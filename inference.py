import os
import torch
import trimesh
import argparse
import time
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from depth2mesh import config
from depth2mesh.checkpoints import CheckpointIO
from depth2mesh.metaavatar import models
import depth2mesh.utils.sdf_meshing as sdf_meshing

from depth2mesh.utils.logs import create_logger

parser = argparse.ArgumentParser(
    description='Inference on test set, extract meshes on novel poses.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite already generated results')
parser.add_argument('--high-res', action='store_true', help='Run marching cubes at high resolution (512^3).')
parser.add_argument('--subject-idx', type=str, default='313',
                    help='Which subject in the validation set to train (and optionally test)')
parser.add_argument('--subsampling-rate', type=int, default=1,
                    help='subsampling rate for sampling training sequences')

# Interpolation
parser.add_argument('--start-offset', type=int, default=0,
                    help='start offset testing sequences')

# Extrapolation
parser.add_argument('--extrapolation', action='store_true',
                    help='Extrapolation on AIST dataset')
parser.add_argument('--aist-sequence', type=str, default='gBR_sBM_cAll_d04_mBR0_ch01',
                    help='AIST sequence name')

parser.add_argument('--exp-suffix', type=str, default='',
                    help='User defined suffix to distinguish different test runs.')

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

    input_type = cfg['data']['input_type']
    vis_n_outputs = cfg['generation']['vis_n_outputs']
    if vis_n_outputs is None:
        vis_n_outputs = -1

    logger, _ = create_logger(generation_dir, phase='test', create_tf_logs=False)

    logger.info('Dataset path: {}'.format(cfg['data']['path']))

    single_view = cfg['data']['single_view']
    dataset_name = cfg['data']['dataset']

    # Model
    model = config.get_model(cfg, device=device)
    ckpt = torch.load(os.path.join(generation_dir, 'test_time_optim.pt'))
    model.decoder.load_state_dict(ckpt['model'])

    # Load forward and backward skinning networks, for novel-pose synthesis
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

    decoder_clone = model.decoder

    subject_idx = 'CoreView_' + args.subject_idx
    if args.extrapolation:
        cfg['data']['dataset'] = 'aist'
        test_dataset = config.get_dataset('test', cfg, subject_idx=subject_idx,
                                          subsampling_rate=args.subsampling_rate,
                                          start_offset=args.start_offset,
                                          sequence=args.aist_sequence)
    else:
        test_dataset = config.get_dataset('test', cfg, subject_idx=subject_idx,
                                          subsampling_rate=args.subsampling_rate, start_offset=args.start_offset)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=0, shuffle=False)

    # Load minimal shape of the target subject, in order to compute bone transformations later
    model_dict = test_dataset.get_model_dict(0)
    gender = model_dict['gender']
    if args.extrapolation:
        minimal_shape = np.load(model_dict['shape_path'])['minimal_shape']
    else:
        minimal_shape = np.load(model_dict['data_path'])['minimal_shape']

    bm_path = os.path.join('./body_models/smpl', gender, 'model.pkl')
    from human_body_prior.body_model.body_model import BodyModel
    bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1, v_template=minimal_shape).cuda()

    # Novel pose synthesis
    model_count = 0
    faces = np.load('body_models/misc/faces.npz')['faces']
    all_skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))


    # Indices of joints for which we set their rotations to 0
    zero_indices = np.array([10, 11, 22, 23])  # feet and hands
    zero_indices_parents = [7, 8, 20, 21]  # and their parents
    # Novel-pose synthesis over test data
    for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        model_count += 1

        # Output folders
        cloth_dir = os.path.join(generation_dir, 'cloth')

        # Get index etc.
        idx = data['idx'].item()

        model_dict = test_dataset.get_model_dict(idx)

        if input_type == 'pointcloud':
            subset = model_dict['subset']
            subject = model_dict['subject']
            data_path = model_dict['data_path']
            filebase = os.path.basename(data_path)[:-4]
        else:
            raise ValueError('Unknown input type: {}'.format(input_type))

        folder_name = os.path.join(subset, subject)
        if args.extrapolation:
            folder_name = os.path.join(folder_name, model_dict['sequence'])

        cloth_dir = os.path.join(cloth_dir, folder_name)

        if not os.path.exists(cloth_dir):
            os.makedirs(cloth_dir)

        poses = data.get('points_corr.pose').to(device)

        colors = np.load('body_models/misc/part_colors.npz')['colors']

        if args.high_res:
            cano_filename = os.path.join(cloth_dir, filebase + '.cano.high')
            posed_filename = os.path.join(cloth_dir, filebase + '.posed.high')
        else:
            cano_filename = os.path.join(cloth_dir, filebase + '.cano')
            posed_filename = os.path.join(cloth_dir, filebase + '.posed')

        rots = data.get('points_corr.rots').to(device)
        Jtrs = data.get('points_corr.Jtrs').to(device)

        # Run grid evaluation and marching-cubes to obtain mesh in canonical space
        if hasattr(decoder_clone, 'hierarchical_pose'):
            if decoder_clone.hierarchical_pose:
                sdf_meshing.create_mesh(decoder_clone,
                                        thetas={'rots': rots, 'Jtrs': Jtrs},
                                        filename=cano_filename, N=512 if args.high_res else 256,
                                        max_batch=32 ** 3)
            else:
                sdf_meshing.create_mesh(decoder_clone,
                                        thetas=poses[0],
                                        filename=cano_filename, N=512 if args.high_res else 256,
                                        max_batch=32 ** 3)
        else:
            sdf_meshing.create_mesh(decoder_clone,
                                    thetas=poses,
                                    filename=cano_filename, N=512 if args.high_res else 256,
                                    max_batch=32 ** 3)

        # Convert canonical pose shape from the normalized space to pointcloud encoder space
        a_pose_trimesh = trimesh.load(cano_filename + '.ply', process=False)

        # Filter out potential floating blobs
        labels = trimesh.graph.connected_component_labels(a_pose_trimesh.face_adjacency)
        components, cnt = np.unique(labels, return_counts=True)
        if len(components) > 1:  # and not args.canonical:
            face_mask = (labels == components[np.argmax(cnt)])
            valid_faces = np.array(a_pose_trimesh.faces)[face_mask, ...]
            n_vertices = len(a_pose_trimesh.vertices)
            vertex_mask = np.isin(np.arange(n_vertices), valid_faces)
            a_pose_trimesh.update_faces(face_mask)
            a_pose_trimesh.update_vertices(vertex_mask)
            # Re-export the processed mesh
            logger.info('Found mesh with floating blobs {}'.format(cano_filename + '.ply'))
            logger.info('Original mesh had {} vertices, reduced to {} vertices after filtering'.format(n_vertices,
                                                                                                       len(a_pose_trimesh.vertices)))
            a_pose_trimesh.export(cano_filename + '.ply')

        # Run forward skinning network on the extracted mesh points
        coord_min = data.get('points_corr.coord_min').to(device)
        coord_max = data.get('points_corr.coord_max').to(device)
        center = data.get('points_corr.center').to(device)

        coord_min = coord_min[0].detach().cpu().numpy()
        coord_max = coord_max[0].detach().cpu().numpy()
        center = center[0].detach().cpu().numpy()

        padding = (coord_max - coord_min) * 0.05
        p_hat_np = (np.array(a_pose_trimesh.vertices) / 2.0 + 0.5) * 1.1 * (
                    coord_max - coord_min) + coord_min - padding + center
        a_pose_trimesh.vertices = p_hat_np
        a_pose_trimesh.export(cano_filename + '.ply')

        p_hat_org = torch.from_numpy(p_hat_np).float().to(device).unsqueeze(0)

        with torch.no_grad():
            coord_max = p_hat_org.max(dim=1, keepdim=True)[0]
            coord_min = p_hat_org.min(dim=1, keepdim=True)[0]

            total_size = (coord_max - coord_min).max(dim=-1, keepdim=True)[0]
            scale = torch.clamp(total_size, min=1.6)
            loc = (coord_max + coord_min) / 2

            sc_factor = 1.0 / scale * 1.5

            p_hat_norm = (p_hat_org - loc) * sc_factor

            # sometimes the point cloud is too dense, we need downsampling to fit into GPU memory
            # num_points = p_hat_norm.shape[1]
            # max_points = 200000
            # if num_points > max_points:
            #     # downsample
            #     indices = torch.randperm(num_points)[:max_points]
            #     p_hat_norm_sample = p_hat_norm[:, indices]
            inp_norm = p_hat_norm

            c = model.encode_inputs(inp_norm, forward=True, scale=scale)

            c_p = model.get_point_features(p_hat_norm, c=c)
            pts_W_fwd = model.decode_w(p_hat_norm, c=c_p, forward=True)
            pts_W_fwd = F.softmax(pts_W_fwd, dim=1).transpose(1, 2)

        skinning_weights_net = pts_W_fwd[0].detach().cpu().numpy()

        # Apply forward LBS to generated posed shape
        trans = data.get('points_corr.trans').cuda()
        root_orient = data.get('points_corr.root_orient').cuda()
        pose_hand = data.get('points_corr.pose_hand').cuda()
        pose_body = data.get('points_corr.pose_body').cuda()
        body = bm(root_orient=root_orient, pose_body=pose_body, pose_hand=pose_hand, trans=trans)
        bone_transforms = body.bone_transforms[0].detach().cpu().numpy()
        Jtr = body.Jtr[0].detach().cpu().numpy()
        Jtr_a_pose = body.Jtr_a_pose[0].detach().cpu().numpy()
        trans = trans[0].detach().cpu().numpy()

        # We set rigid transforms of the hands and feet to be the same as their parents
        # as they are often not accurately registered
        bone_transforms[zero_indices, ...] = bone_transforms[zero_indices_parents, ...]

        T = np.dot(skinning_weights_net, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        # Compute T such that it transforms points in Vitruvian A-pose to transformed space
        bone_transforms_02v = get_transforms_02v(Jtr_a_pose)
        T_v = np.dot(skinning_weights_net, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        T = np.matmul(T, np.linalg.inv(T_v))

        # Transform mesh points
        n_pts = p_hat_np.shape[0]
        homogen_coord = np.ones([n_pts, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([p_hat_np, homogen_coord], axis=-1).reshape([n_pts, 4, 1])
        body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans

        # Create and save transformed mesh
        posed_trimesh = trimesh.Trimesh(vertices=body_mesh, faces=a_pose_trimesh.faces, process=False)
        posed_trimesh.visual = a_pose_trimesh.visual
        posed_trimesh.export(posed_filename + '.ply')
        # np.save(os.path.join(cloth_dir, filebase + '.pelvis.npy'), Jtr[0])
        logger.info("Exported mesh: {}".format(posed_filename + '.ply'))