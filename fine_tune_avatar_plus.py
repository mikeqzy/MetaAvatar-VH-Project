import os
import torch
import trimesh
import argparse
import time
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from depth2mesh import config
from depth2mesh.checkpoints import CheckpointIO
from depth2mesh.metaavatar import models

from depth2mesh.utils.logs import create_logger

parser = argparse.ArgumentParser(
    description='Do fine-tuning on validation set, then extract meshes on novel poses.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite already generated results')
parser.add_argument('--subsampling-rate', type=int, default=1,
                    help='subsampling rate for sampling training sequences')
parser.add_argument('--epochs-per-run', type=int, default=-1,
                    help='Number of epochs to train before restart.')
parser.add_argument('--optim-epochs', type=int, default=-1,
                    help='Number of total epochs  to train.')
parser.add_argument('--num-workers', type=int, default=8,
                    help='Number of workers to use for train and val loaders.')
parser.add_argument('--subject-idx', type=str, default='313',
                    help='Which subject in the validation set to train (and optionally test)')
parser.add_argument('--exclude-hand', action='store_true',
                    help='Replace reconstructed hand with smpl hand')
parser.add_argument('--use-normal', action='store_true',
                    help='Use normal loss')

parser.add_argument('--exp-suffix', type=str, default='',
                    help='User defined suffix to distinguish different test runs.')

def get_skinning_weights(pts, src, ref_W):
    """
    Finds skinning weights of pts on src via barycentric interpolation.
    """
    closest_face, closest_points = src.closest_faces_and_points(pts)
    vert_ids, bary_coords = src.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
    pts_W = (ref_W[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)

    return pts_W

def compute_sdf_loss(model_output, gt):
    loss_dict = sdf_loss(model_output, gt)
    total_loss = torch.zeros(1, device=device)
    for loss_name, loss in loss_dict.items():
        total_loss += loss.mean()

    return total_loss, loss_dict

def mask_by_reproj_dist(p, p_rp, mode='mean', value=-1):
    if mode == 'mean':
        thr = torch.norm(p - p_rp, dim=-1).mean(-1, keepdim=True)
    else:
        thr = value

    mask = (torch.norm(p - p_rp, dim=-1) < thr).unsqueeze(-1)

    return mask

def normalize_canonical_points(pts, coord_min, coord_max, center):
    pts -= center
    padding = (coord_max - coord_min) * 0.05
    pts = (pts - coord_min + padding) / (coord_max - coord_min) / 1.1
    pts -= 0.5
    pts *= 2.

    return pts

def get_transforms_02v(Jtr):
    from scipy.spatial.transform import Rotation as R
    rot45p = R.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = R.from_euler('z', -45, degrees=True).as_matrix()
    # Specify the bone transformations that transform a SMPL A-pose mesh
    # to a star-shaped A-pose (i.e. Vitruvian A-pose)
    bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))
    # Jtr *= sc_factor

    # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
    chain = [1, 4, 7, 10]
    rot = rot45p.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)
    # Second chain: R-hip (2), R-knee (5), R-ankle (8), R-foot (11)
    chain = [2, 5, 8, 11]
    rot = rot45n.copy()
    for i, j_idx in enumerate(chain):
        bone_transforms_02v[j_idx, :3, :3] = rot
        t = Jtr[j_idx].copy()
        if i > 0:
            parent = chain[i-1]
            t_p = Jtr[parent].copy()
            t = np.dot(rot, t - t_p)
            t += bone_transforms_02v[parent, :3, -1].copy()

        bone_transforms_02v[j_idx, :3, -1] = t

    bone_transforms_02v[chain, :3, -1] -= np.dot(Jtr[chain], rot.T)

    return bone_transforms_02v

if __name__ == '__main__':
    args = parser.parse_args()
    cfg = config.load_config(args.config, 'configs/default.yaml')
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
    generation_dir += args.exp_suffix
    out_time_file = os.path.join(generation_dir, 'time_generation_full.pkl')
    out_time_file_class = os.path.join(generation_dir, 'time_generation.pkl')
    stage = cfg['training']['stage']
    inner_lr = cfg['training']['inner_lr']

    batch_size = cfg['training']['inner_batch_size']
    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1

    logger, _ = create_logger(generation_dir, phase='test', create_tf_logs=False)

    logger.info('Dataset path: {}'.format(cfg['data']['path']))

    single_view = cfg['data']['single_view']
    dataset_name = cfg['data']['dataset']

    subject_idx = 'CoreView_' + args.subject_idx

    train_dataset = config.get_dataset('test', cfg, subsampling_rate=args.subsampling_rate, subject_idx=subject_idx,
                                       exclude_hand=args.exclude_hand)

    # Loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True)

    # Model
    model = config.get_model(cfg, device=device, dataset=train_dataset)
    ckpt = torch.load(os.path.join(out_dir, cfg['test']['model_file']))
    decoder_state_dict = OrderedDict()

    # Load meta-learned SDF decoder
    for k, v in ckpt['model'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('decoder'):
            decoder_state_dict[k[8:]] = v

    model.decoder.load_state_dict(decoder_state_dict)

    # Load forward and backward skinning networks, for fine-tuning
    optim_skinning_net_path = cfg['model']['skinning_net2']
    ckpt = torch.load(optim_skinning_net_path)

    encoder_fwd_state_dict = OrderedDict()
    skinning_decoder_fwd_state_dict = OrderedDict()
    encoder_bwd_state_dict = OrderedDict()
    skinning_decoder_bwd_state_dict = OrderedDict()
    for k, v in ckpt['model'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('skinning_decoder_fwd'):
            skinning_decoder_fwd_state_dict[k[21:]] = v
        elif k.startswith('skinning_decoder_bwd'):
            skinning_decoder_bwd_state_dict[k[21:]] = v
        elif k.startswith('encoder_fwd'):
            encoder_fwd_state_dict[k[12:]] = v
        elif k.startswith('encoder_bwd'):
            encoder_bwd_state_dict[k[12:]] = v

    model.encoder_fwd.load_state_dict(encoder_fwd_state_dict)
    model.encoder_bwd.load_state_dict(encoder_bwd_state_dict)
    model.skinning_decoder_fwd.load_state_dict(skinning_decoder_fwd_state_dict)
    model.skinning_decoder_bwd.load_state_dict(skinning_decoder_bwd_state_dict)

    model.eval()
    import depth2mesh.utils.sdf_meshing as sdf_meshing
    from depth2mesh.utils.loss_functions import sdf_with_mask as sdf_loss

    # Create a clone of meta-learned SDF decoder
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    decoder_clone = models.decoder_dict[decoder](**decoder_kwargs)
    decoder_clone.load_state_dict(model.decoder.state_dict())
    decoder_clone = decoder_clone.to(device)

    if stage == 'meta-hyper' and cfg['model']['decoder'] == 'hyper_bvp':
        if model.decoder.hierarchical_pose:
            inner_optimizer = torch.optim.Adam(
                params = [
                    {
                        "params": decoder_clone.net.parameters(),
                        "lr": inner_lr,
                    },
                    {
                        "params": decoder_clone.pose_encoder.parameters(),
                        "lr": 1e-4,
                    }
                ]
            )
        else:
            inner_optimizer = torch.optim.Adam(decoder_clone.parameters(), lr=inner_lr)
    else:
        raise ValueError('Fine-tuning only supports meta-hyper stage \
                          with SDF decoder type hyper_bvp. Got stage {} and SDF \
                          decoder {}'.format(stage, cfg['model']['decoder']))

    # Checkpoint for fine-tuned SDF decoder
    test_optim_ckpt_io = CheckpointIO(generation_dir, model=decoder_clone, optimizer=inner_optimizer)
    test_optim_ckpt_filename = 'test_time_optim.pt'
    logger.info(test_optim_ckpt_filename)
    try:
        load_dict = test_optim_ckpt_io.load(test_optim_ckpt_filename)
    except FileExistsError:
        load_dict = dict()

    epoch_it = load_dict.get('epoch_it', -1)
    proj_thr = cfg['training']['proj_thr']  # re-projection threshold to filter out invalid points mapped by backward LBS

    if args.optim_epochs > 0:
        max_epoch = args.optim_epochs
    else:
        max_epoch = cfg['test']['optim_iterations']

    # Time statistics
    time_dict = OrderedDict()
    time_dict['network_time'] = 0

    # Fine-tuning loop
    epoch_cnt = 0
    epochs_to_run = args.epochs_per_run if args.epochs_per_run > 0 else (max_epoch + 1)
    for _ in range(epochs_to_run):
        epoch_it += 1
        if epoch_it >= max_epoch:
            break

        for idx, data in enumerate(train_loader):
            inputs = data.get('inputs').to(device)
            points_corr = data.get('points_corr').to(device)
            poses = data.get('points_corr.pose').to(device)

            scale = data.get('points_corr.scale').to(device)
            scale = scale.view(-1, 1, 1)
            bone_transforms = data.get('points_corr.bone_transforms').to(device)
            bone_transforms_02v = data.get('points_corr.bone_transforms_02v').to(device)
            minimal_shape = data.get('points_corr.minimal_shape').to(device)
            normals = data.get('points_corr.normals').to(device)
            # TODO: pass normal into kwargs
            kwargs = {'scale': scale, 'bone_transforms': bone_transforms,
                      'bone_transforms_02v': bone_transforms_02v,
                      'minimal_shape': minimal_shape,
                      'normals': None,
                      }
            if args.use_normal:
                kwargs.update(normals=normals)

            # TODO: we should get rid of this by re-calculating center by bounding volume
            # not mean of points
            coord_min = data.get('points_corr.coord_min').to(device).view(-1, 1, 1)
            coord_max = data.get('points_corr.coord_max').to(device).view(-1, 1, 1)
            center = data.get('points_corr.center').to(device).unsqueeze(1)

            # Use the learned skinning net to transform points to A-pose
            t = time.time()
            with torch.no_grad():
                out_dict = model(inputs, points_corr, stage='skinning_weights', **kwargs)

            points_corr_hat = out_dict.get('p_hat')
            points_corr_reproj = out_dict.get('p_rp')
            normals_a_pose = out_dict.get('normals_a_pose')

            # Do the following:
            # 1) Filter out points whose re-projection distance is greater than the specified threshold
            # 2) Normalize valid points to [-1, 1]^3 for SDF decoder
            mask = mask_by_reproj_dist(points_corr, points_corr_reproj, mode='constant', value=proj_thr)

            points_corr_hat = points_corr_hat * scale / 1.5
            points_corr_hat = normalize_canonical_points(points_corr_hat, coord_min=coord_min, coord_max=coord_max, center=center)

            batch_size = points_corr_hat.size(0)

            # Generate point samples for fine-tuning
            on_surface_samples = points_corr_hat.size(1)
            off_surface_samples = on_surface_samples
            total_samples = on_surface_samples + off_surface_samples

            on_surface_coords = points_corr_hat
            on_surface_normals = normals_a_pose

            off_surface_coords = (torch.rand(batch_size, off_surface_samples, 3, device=device, dtype=torch.float32) - 0.5) * 2
            off_surface_normals = torch.ones(batch_size, off_surface_samples, 3, device=device, dtype=torch.float32) * -1

            sdf = torch.zeros(batch_size, total_samples, 1, device=device, dtype=torch.float32)  # on-surface = 0
            sdf[:, on_surface_samples:, :] = -1  # off-surface = -1

            coords_in = torch.cat([on_surface_coords, off_surface_coords], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask)], dim=1)

            # Use normal information if available.
            if on_surface_normals is not None:
                normals_in = torch.cat([on_surface_normals, off_surface_normals], dim=1)
            else:
                normals_in = torch.zeros_like(coords_in)

            decoder_input = {'coords': coords_in}
            if decoder_clone.hierarchical_pose:
                rots = data.get('points_corr.rots').to(device)
                Jtrs = data.get('points_corr.Jtrs').to(device)
                decoder_input.update({'rots': rots, 'Jtrs': Jtrs})
            else:
                decoder_input.update({'cond': poses})

            gt = {'sdf': sdf, 'normals': normals_in, 'mask': mask}

            # Forward pass and compute loss
            inner_output = decoder_clone(decoder_input)
            inner_loss, inner_loss_dict = compute_sdf_loss(inner_output, gt)

            # Regularize on predicted SDF parameters
            params = torch.cat(inner_output['params'], dim=1)
            n_params = params.size(-1)
            inner_loss += params.norm(dim=-1).mean() * 1e2 / n_params

            # Do one step of optimization
            decoder_clone.zero_grad()
            inner_loss.backward()
            inner_optimizer.step()

            # Update timing
            time_dict['network_time'] += time.time() - t

            # Logging
            log_str = 'Epoch {}: '.format(epoch_it)
            for k, v in inner_loss_dict.items():
                log_str += '{} loss: {:.4f},'.format(k, v.item())
            log_str += f'Total loss: {inner_loss.item():.4f}'

            logger.info(log_str)

        epoch_cnt += 1

    logger.info('Elapsed network time: {} seconds.'.format(time_dict['network_time']))

    # exit(0)

    # Save fine-tuned model
    if epoch_cnt > 0:
        test_optim_ckpt_io.save(test_optim_ckpt_filename, epoch_it=epoch_it)

    # If we have not reached desired fine-tuning epoch, then exit with code 3.
    # This for job-chaining on HPC clusters. You can ignore this if you run
    # fine-tuning on local machines.
    if epoch_it < max_epoch:
        exit(3)

    # Novel pose synthesis
    # Should be similar to fine_tune_avatar.py
    # But customized dataloaders for arbitray pose dataset can take sometime to implement