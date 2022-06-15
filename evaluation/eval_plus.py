"""
Code for evaluation. Copied and modified from https://github.com/bharat-b7/IPNet
"""
import sys
sys.path.append('/local/home/zhqian/code/MetaAvatar-release')

import os
import torch
import trimesh
import argparse
import numpy as np
import kaolin as kal
from kaolin.rep import TriangleMesh as tm
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial.distance import cdist

from evaluation.lib.mesh_distance import chamfer_l2_distance, point_to_surface_vec, normal_consistency_vertex
from depth2mesh import config, data
from depth2mesh.utils.logs import create_logger

parser = argparse.ArgumentParser(
    description='Inference on test set, extract meshes on novel poses.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite already generated results')
parser.add_argument('--subsampling-rate', type=int, default=1,
                    help='subsampling rate for sampling training sequences')
parser.add_argument('--start-offset', type=int, default=0,
                    help='start offset testing sequences')
parser.add_argument('--bi-directional', action='store_true', help='Whether to evaluate bi-directional distances or not.')
parser.add_argument('--subject-idx', type=str, default='313',
                    help='Which subject in the validation set to train (and optionally test)')
parser.add_argument('--high-res', action='store_true', help='Run marching cubes at high resolution (512^3).')

parser.add_argument('--exp-suffix', type=str, default='',
                    help='User defined suffix to distinguish different test runs.')

smpl_faces = np.load('body_models/misc/faces.npz')['faces']
skinning_weights = np.load('body_models/misc/skinning_weights_all.npz')
body_indices = np.array([0, 1, 2, 3, 4, 5, 6, 9, 13, 14, 16, 17, 18, 19], dtype=np.int64)

def evaluate_CD_NC(args):
    cfg = config.load_config(args.config, 'configs/default.yaml')
    out_dir = cfg['training']['out_dir']
    generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir']) + args.exp_suffix
    is_cuda = (torch.cuda.is_available() and not args.no_cuda)
    device = torch.device("cuda" if is_cuda else "cpu")

    # cloth_split = [v for v in args.cloth_split.split(',')]
    # act_split = [v for v in args.act_split.split(',')]

    phase = 'chamfer_dist_normal_consistency_subject{}'.format(args.subject_idx)

    if args.bi_directional:
        phase += '_bi'

    logger, _ = create_logger(generation_dir, phase=phase, create_tf_logs=False)

    subject_idx = 'CoreView_' + args.subject_idx
    dataset = config.get_dataset('test', cfg, subject_idx=subject_idx,
                                      subsampling_rate=args.subsampling_rate, start_offset=args.start_offset)

    logger.info(len(dataset.data))
    batch_size = 1

    # Loader
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=False)

    dists = []
    s_dists = []

    nc_vs = []

    for it, data in enumerate(tqdm(test_loader)):
        idxs = data['idx'].cpu().numpy()

        # Directories to load meshes
        mesh_dir = os.path.join(generation_dir, 'cloth')

        for i, idx in enumerate(idxs):
            model_dict = dataset.get_model_dict(idx)

            subset = model_dict['subset']
            subject = model_dict['subject']
            gender = model_dict['gender']
            scan_path = model_dict['scan_path']
            filebase = os.path.basename(model_dict['data_path'])[:-4]

            folder_name = os.path.join(subset, subject)

            mesh_dir_ = os.path.join(mesh_dir, folder_name)

            # Load generated meshes
            if args.high_res:
                mesh_file_name = filebase + '.posed.high.ply'
            else:
                mesh_file_name = filebase + '.posed.ply'
            posed_mesh_name = os.path.join(mesh_dir_, mesh_file_name)
            posed_trimesh = trimesh.load(posed_mesh_name, process=False)
            posed_vertices_np = np.array(posed_trimesh.vertices, dtype=np.float32)
            posed_vertices = torch.tensor(posed_vertices_np, requires_grad=False, device=device)
            posed_faces = np.array(posed_trimesh.faces, dtype=np.int64)
            posed_tm = tm.from_tensors(vertices=posed_vertices,
                                       faces=torch.tensor(posed_faces, requires_grad=False, device=device))

            gt_trimesh = trimesh.load(scan_path)
            if np.max(gt_trimesh.vertices) > 10:
                gt_trimesh.vertices /= 1000 # mm to m
            gt_vertices_np = np.array(gt_trimesh.vertices, dtype=np.float32)
            gt_vertices = torch.tensor(gt_vertices_np, requires_grad=False, device=device)
            gt_faces = np.array(gt_trimesh.faces, dtype=np.int64)
            gt_tm = tm.from_tensors(vertices=gt_vertices,
                                    faces=torch.tensor(gt_faces.astype(np.int64), requires_grad=False, device=device))

            # breakpoint()
            smpl_file = model_dict['data_path'][:-4] + '.ply'
            smpl_trimesh = trimesh.load(smpl_file, process=False)
            smpl_vertices = np.array(smpl_trimesh.vertices, dtype=np.float32)

            closest_vertice_posed = np.argmin(cdist(posed_vertices_np, smpl_vertices), axis=1)
            closest_vertice_gt = np.argmin(cdist(gt_vertices_np, smpl_vertices), axis=1)

            # Find labels that are neither feet nor hands
            smpl_labels = skinning_weights[gender].argmax(-1)
            smpl_mask = np.isin(smpl_labels, body_indices)
            smpl_inds = np.where(smpl_mask)[0]

            # part_mask = np.full((gt_vertices.shape[0],), True)
            part_mask = np.isin(closest_vertice_gt, smpl_inds)

            dist, dist2gt, dist2mesh, closest_index_in_gt, closest_index_in_mesh = chamfer_l2_distance(posed_vertices.unsqueeze(0) * 100, gt_vertices.unsqueeze(0) * 100, w1=0.5, w2=0.5)
            if args.bi_directional:
                # mesh_mask = np.isin(closest_index_in_gt[0].detach().cpu().numpy(), part_inds)
                # mesh_mask = np.full((posed_vertices.shape[0],), True)
                mesh_mask = np.isin(closest_vertice_posed, smpl_inds)

            if args.bi_directional:
                dist = (dist2mesh[:, part_mask].mean() * 0.5 + dist2gt[:, mesh_mask].mean() * 0.5).item()
            else:
                dist = dist2mesh[:, part_mask].mean().item()

            if not np.isnan(dist):
                dists.append(dist)

            with torch.no_grad():
                gt_vertices_masked = gt_vertices[part_mask, :].clone()
                g2p = point_to_surface_vec(gt_vertices_masked, posed_tm)
                if args.bi_directional:
                    posed_vertices_masked = posed_vertices[mesh_mask, :].clone()
                    p2g = point_to_surface_vec(posed_vertices_masked, gt_tm)
                    s_dist = ((p2g.sqrt().mean() + g2p.sqrt().mean()) * 100 / 2).item()
                else:
                    s_dist = (g2p.sqrt().mean() * 100).item()

            if not np.isnan(s_dist):
                s_dists.append(s_dist)

            nc_v = normal_consistency_vertex(posed_trimesh, gt_trimesh, part_mask)
            if not np.isnan(nc_v):
                nc_vs.append(nc_v)

            logger.info('Chamfer distance for input {}: {} cm'.format(filebase, dist))
            logger.info('Mesh distance for input {}: {} cm'.format(filebase, s_dist))
            logger.info('Vertex normal consistency for input {}: {}'.format(filebase, nc_v))

    logger.info('Mean Chamfer distance: {} cm'.format(np.mean(dists)))
    logger.info('Mean mesh distance: {} cm'.format(np.mean(s_dists)))
    logger.info('Mean vertex normal consistency: {} cm'.format(np.mean(nc_vs)))

    logger.info('Max Chamfer distance: {} cm'.format(np.max(dists)))
    logger.info('Max mesh distance: {} cm'.format(np.max(s_dists)))
    logger.info('Min vertex normal consistency: {} cm'.format(np.min(nc_vs)))

    logger.info('Median Chamfer distance: {} cm'.format(np.median(dists)))
    logger.info('Median mesh distance: {} cm'.format(np.median(s_dists)))
    logger.info('Median vertex normal consistency: {} cm'.format(np.median(nc_vs)))


def main(args):
    evaluate_CD_NC(args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
