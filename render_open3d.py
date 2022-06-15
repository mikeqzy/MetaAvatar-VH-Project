import argparse
import os
from os.path import join
from glob import glob
import trimesh
import numpy as np
import imageio
import open3d as o3d

parser = argparse.ArgumentParser(description='Render generated mesh')
parser.add_argument('--exp-suffix', type=str, default='')
parser.add_argument('--subject-idx', type=str, default='313')
parser.add_argument('--aist-sequence', type=str, default='gBR_sBM_cAll_d04_mBR0_ch01')

def render_single_mesh(mesh_file, save_file):
    body_mesh = o3d.io.read_triangle_mesh(mesh_file)
    body_mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(body_mesh)
    vis.update_geometry(body_mesh)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_file)


if __name__ == '__main__':
    args = parser.parse_args()
    root = '/local/home/zhqian/code/MetaAvatar-release/out/meta-avatar/' \
           'conv-unet-plane64x3_ZJU-MoCap_keep-aspect_stage2-inner-1e-6-outer-1e-6_1gpus/'
    root = join(root, f'generation{args.exp_suffix}', 'cloth', 'aist++',
                f'CoreView_{args.subject_idx}', args.aist_sequence)
    mesh_files = sorted(glob(join(root, '*.posed.high.ply')))
    images = []
    body_mesh = o3d.io.read_triangle_mesh(mesh_files[6])
    body_mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(body_mesh)
    for mesh_file in mesh_files:
        body_mesh_ = o3d.io.read_triangle_mesh(mesh_file)
        body_mesh_.compute_vertex_normals()

        body_mesh.vertices = body_mesh_.vertices
        body_mesh.triangles = body_mesh_.triangles
        body_mesh.vertex_normals = body_mesh_.vertex_normals
        body_mesh.triangle_normals = body_mesh_.triangle_normals
        # body_mesh.vertex_colors = body_mesh_.vertex_colors

        vis.update_geometry(body_mesh)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(join('tmp.png'))
        image = imageio.imread('tmp.png')
        images.append(image)
    os.remove('tmp.png')
    imageio.mimwrite(join(root, 'demo_open3d.mp4'), images, fps=5)