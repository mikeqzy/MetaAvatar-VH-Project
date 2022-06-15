import os
import glob
import numpy as np
import trimesh
import igl
import pickle as pkl

from torch.utils import data
import torch

from scipy.spatial.transform import Rotation as R

from human_body_prior.body_model.body_model import BodyModel

class AISTDataset(data.Dataset):
    ''' Raw scan dataset class.
    '''

    def __init__(self, subject_folder='/local/home/zhqian/data/ZJU-MoCap',
                 subjects=('CoreView_313',),
                 motion_root='/local/home/zhqian/data/AIST++',
                 sequence='gBR_sBM_cAll_d04_mBR0_ch01',
                 mode='test',
                 normalized_scale=True,
                 subsampling_rate=1,
                 keep_aspect_ratio=True
                 ):
        ''' Initialization of the CAPECorrDataset instance.

        Args:
            dataset_folder (str): folder that stores processed, registered models
            subjects (list of strs): which subjects to include in this dataset instance
            mode (str): mode of the dataset. Can be either 'train', 'val' or 'test'
            input_pointcloud_n (int): number for points to sample from each scan
            normalized_scale (bool): normalize all points into [-1, 1]
            subsampling_rate (int): subsampling rate for subsampling dataset frames
            start_offset (int): first index for sampling the dataset
            keep_aspect_ratio (bool): whether to keep aspect ratio when normalizing canonical space
        '''
        # Attributes
        self.subject_folder = subject_folder
        self.mode = mode
        self.normalized_scale = normalized_scale

        self.faces = np.load('body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('body_models/misc/J_regressors.npz'))

        self.v_templates = dict(np.load('body_models/misc/v_templates.npz'))

        self.keep_aspect_ratio = keep_aspect_ratio

        self.rot45p = R.from_euler('z', 45, degrees=True).as_matrix()
        self.rot45n = R.from_euler('z', -45, degrees=True).as_matrix()

        # extract minimal shape
        subject = subjects[0]
        shape_folder = os.path.join(subject_folder, subject, 'models')
        shape_file = os.path.join(shape_folder, '000001.npz')
        # minimal_shape = np.load(model_file)['minimal_shape']
        gender = 'neutral'

        bm_path = os.path.join('./body_models/smpl', gender, 'model.pkl')
        bm = BodyModel(bm_path=bm_path, num_betas=10, batch_size=1).cuda()
        self.bm = bm

        # Get all data
        self.data = []

        subset = 'aist++'

        input_file = os.path.join(motion_root, sequence + '.pkl')
        with open(input_file, 'rb') as f:
            input_data = pkl.load(f)
        poses = input_data['smpl_poses'][::subsampling_rate]
        transl = input_data['smpl_trans'][::subsampling_rate] / 100.

        for cnt, (pose, trans) in enumerate(zip(poses, transl)):
            idx = cnt * subsampling_rate

            self.data.append(
                    {'subset': subset,
                     'subject': subject,
                     'sequence': sequence,
                     'gender': gender,
                     'frame_idx': idx,
                     'data_path': f'dummy/{idx:06d}.npz',
                     'pose': pose,
                     'trans': trans,
                     'shape_path': shape_file}
                    )

    def augm_params(self, roll_range=10, pitch_range=180, yaw_range=10):
        """ Get augmentation parameters.

        Args:
            roll_range (int): roll angle sampling range (train mode) or value (test mode)
            pitch_range (int): pitch angle sampling range (train mode) or value (test mode)
            yaw_range (int): yaw angle sampling range (train mode) or value (test mode)

        Returns:
            rot_mat (4 x 4 float numpy array): homogeneous rotation augmentation matrix.
        """
        if self.mode == 'train':
            # Augmentation during training

            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            # Roll
            rot_x = min(2*roll_range,
                    max(-2*roll_range, np.random.randn()*roll_range))

            sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
            rot_x = np.eye(4)
            rot_x[1, 1] = cs
            rot_x[1, 2] = -sn
            rot_x[2, 1] = sn
            rot_x[2, 2] = cs
            # but it is identity with probability 3/5
            if np.random.uniform() <= 0.6:
                rot_x = np.eye(4)

            rot_y = min(2*pitch_range,
                    max(-2*pitch_range, (np.random.rand() * 2 - 1)*pitch_range))

            # Pitch
            sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
            rot_y = np.eye(4)
            rot_y[0, 0] = cs
            rot_y[0, 2] = sn
            rot_y[2, 0] = -sn
            rot_y[2, 2] = cs

            rot_z = min(2*yaw_range,
                    max(-2*yaw_range, np.random.randn()*yaw_range))

            # Yaw
            sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
            rot_z = np.eye(4)
            rot_z[0, 0] = cs
            rot_z[0, 1] = -sn
            rot_z[1, 0] = sn
            rot_z[1, 1] = cs
            # but it is identity with probability 3/5
            if np.random.uniform() <= 0.6:
                rot_z = np.eye(4)

            rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))
        elif self.mode in ['test']:
            # Simulate a rotating camera

            # Roll
            rot_x = roll_range

            sn, cs = np.sin(np.pi / 180 * rot_x), np.cos(np.pi / 180 * rot_x)
            rot_x = np.eye(4)
            rot_x[1, 1] = cs
            rot_x[1, 2] = -sn
            rot_x[2, 1] = sn
            rot_x[2, 2] = cs

            rot_y = pitch_range

            # Pitch
            sn, cs = np.sin(np.pi / 180 * rot_y), np.cos(np.pi / 180 * rot_y)
            rot_y = np.eye(4)
            rot_y[0, 0] = cs
            rot_y[0, 2] = sn
            rot_y[2, 0] = -sn
            rot_y[2, 2] = cs

            rot_z = yaw_range

            # Yaw
            sn, cs = np.sin(np.pi / 180 * rot_z), np.cos(np.pi / 180 * rot_z)
            rot_z = np.eye(4)
            rot_z[0, 0] = cs
            rot_z[0, 1] = -sn
            rot_z[1, 0] = sn
            rot_z[1, 1] = cs

            rot_mat = np.dot(rot_x, np.dot(rot_y, rot_z))
        else:
            # No augmentation
            rot_mat = np.eye(4)

        return rot_mat

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        shape_file = self.data[idx]['shape_path']
        shape_file = np.load(shape_file)
        minimal_shape = shape_file['minimal_shape']
        betas = shape_file['betas']

        pose = self.data[idx]['pose']
        trans = self.data[idx]['trans']
        subject = self.data[idx]['subject']
        gender = self.data[idx]['gender']

        data = {}

        # compute bone transforms
        pose = pose.astype(np.float32)
        root_orient = pose[:3].copy()
        pose_body = pose[3:66].copy()
        pose_hand = pose[66:].copy()
        trans = trans.astype(np.float32)

        root_orient_torch = torch.from_numpy(root_orient).unsqueeze(0).cuda()
        pose_body_torch = torch.from_numpy(pose_body).unsqueeze(0).cuda()
        pose_hand_torch = torch.from_numpy(pose_hand).unsqueeze(0).cuda()
        betas_torch = torch.from_numpy(betas).cuda()
        trans_torch = torch.from_numpy(trans).unsqueeze(0).cuda()

        body = self.bm(root_orient=root_orient_torch, pose_body=pose_body_torch, pose_hand=pose_hand_torch,
                  betas=betas_torch, trans=trans_torch)
        bone_transforms_org = body.bone_transforms.detach().cpu().numpy()[0]
        bone_transforms = bone_transforms_org.copy()

        # We set aug_rot to be inverse of global rotation, such that point clouds will always have 0 global orientation
        # otherwise backward skinning net does not work
        aug_rot = bone_transforms_org[0].copy().T
        aug_rot[-1, :3] = 0

        # Also get GT SMPL poses
        pose = np.concatenate([pose_body, pose_hand], axis=-1)

        pose = R.from_rotvec(pose.reshape([-1, 3]))
        pose_quat = pose.as_quat()

        pose_quat = pose_quat.reshape(-1)
        pose_rot = np.concatenate([np.expand_dims(np.eye(3), axis=0), pose.as_matrix()], axis=0).reshape([-1, 9])   # 24 x 9

        # Minimally clothed shape
        # minimal_shape_path = os.path.join(self.cape_path, 'cape_release', 'minimal_body_shape', subject, subject + '_minimal.npy')
        posedir = self.posedirs[gender]
        J_regressor = self.J_regressor[gender]
        # minimal_shape = np.load(minimal_shape_path)
        Jtr = np.dot(J_regressor, minimal_shape)

        n_smpl_points = minimal_shape.shape[0]

        pose_mat = pose.as_matrix()
        ident = np.eye(3)
        pose_feature = (pose_mat - ident).reshape([207, 1])
        pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
        minimal_shape += pose_offsets

        # Get posed clothed and minimally-clothed SMPL meshes
        skinning_weights = self.skinning_weights[gender]
        T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])

        homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)

        a_pose_homo = np.concatenate([minimal_shape, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        minimal_body_mesh = np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans

        # Apply rotation augmentation
        center = minimal_body_mesh.mean(0)


        bone_transforms[:, :3, -1] += trans - center
        bone_transforms = np.matmul(np.expand_dims(aug_rot, axis=0), bone_transforms)
        bone_transforms[:, :3, -1] += center


        # Specify the bone transformations that transform a SMPL A-pose mesh
        # to a star-shaped A-pose (i.e. Vitruvian A-pose)
        bone_transforms_02v = np.tile(np.eye(4), (24, 1, 1))

        # First chain: L-hip (1), L-knee (4), L-ankle (7), L-foot (10)
        chain = [1, 4, 7, 10]
        rot = self.rot45p.copy()
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
        rot = self.rot45n.copy()
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
        bone_transforms_02v_org = bone_transforms_02v.copy()

        T = np.matmul(skinning_weights, bone_transforms_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        minimal_shape_v = np.matmul(T[:, :3, :3], minimal_shape[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        # Normalize conanical pose points with GT full-body scales. This should be fine as
        # at test time we register each frame first, thus obtaining full-body scale
        center = np.mean(minimal_shape_v, axis=0)
        # center = np.zeros(3, dtype=np.float32)
        minimal_shape_v_centered = minimal_shape_v - center
        if self.keep_aspect_ratio:
            coord_max = minimal_shape_v_centered.max()
            coord_min = minimal_shape_v_centered.min()
        else:
            coord_max = minimal_shape_v_centered.max(axis=0, keepdims=True)
            coord_min = minimal_shape_v_centered.min(axis=0, keepdims=True)

        padding = (coord_max - coord_min) * 0.05

        Jtr_norm = Jtr - center
        Jtr_norm = (Jtr_norm - coord_min + padding) / (coord_max - coord_min) / 1.1
        Jtr_norm -= 0.5
        Jtr_norm *= 2.


        data = {
            'trans': trans.astype(np.float32),
            'pose': pose_quat.astype(np.float32),
            'bone_transforms': bone_transforms.astype(np.float32),
            'bone_transforms_org': bone_transforms_org.astype(np.float32),
            'bone_transforms_02v': bone_transforms_02v.astype(np.float32),
            'bone_transforms_02v_org': bone_transforms_02v_org.astype(np.float32),
            'coord_max': coord_max.astype(np.float32),
            'coord_min': coord_min.astype(np.float32),
            'center': center.astype(np.float32),
            'minimal_shape': minimal_shape_v.astype(np.float32),
            'root_orient': root_orient,
            'pose_hand': pose_hand,
            'pose_body': pose_body,
            'rots': pose_rot.astype(np.float32),
            'Jtrs': Jtr_norm.astype(np.float32),
        }

        data_out = {}
        field_name = 'points_corr'
        for k, v in data.items():
            if k is None:
                data_out[field_name] = v
            else:
                data_out['%s.%s' % (field_name, k)] = v

            data_out.update({'gender': gender,
                             'idx': idx,})

        return data_out

    def get_model_dict(self, idx):
        return self.data[idx]
