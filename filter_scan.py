import os.path

import trimesh
import numpy as np
from glob import glob
from os.path import join

if __name__ == '__main__':
    plane_normal = np.array([0.,
                             0.,
                             1.])
    plane_origin = np.array([0., 0., 0.01])
    path = '/local/home/zhqian/data/ZJU-MoCap/CoreView_315/raw_scans'
    meshes = glob(join(path, '*'))
    for mesh_file in meshes:
        mesh = trimesh.load(mesh_file)
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        components, cnt = np.unique(labels, return_counts=True)
        if len(components) > 1:  # and not args.canonical:
            print(mesh_file)
            face_mask = (labels == components[np.argmax(cnt)])
            valid_faces = np.array(mesh.faces)[face_mask, ...]
            n_vertices = len(mesh.vertices)
            vertex_mask = np.isin(np.arange(n_vertices), valid_faces)
            mesh.update_faces(face_mask)
            mesh.update_vertices(vertex_mask)

        new_mesh = mesh.slice_plane(plane_origin, plane_normal, cap=True)
        # Re-export the processed mesh
        new_mesh.export(join(path, '..', 'scans', os.path.basename(mesh_file)))