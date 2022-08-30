from black import out
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transforms3d.quaternions import quat2mat
import open3d as o3d
import trimesh
import glob
import os
import sys
from copy import deepcopy
# TODO: do this better
sys.path.append('..')
from src.third_party.ConvONets.utils.libmesh import check_mesh_contains

def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def T_inverse_with_scaling(T, with_scale=False):
    inverse_T = np.eye(4)

    sR = T[:3, :3]
    scale_sq = np.transpose(sR) @ sR
    scale = np.sqrt(scale_sq[0,0])
    R = sR / scale
    t = T[:3, 3]

    s_inverse = 1 / scale
    R_inverse = np.transpose(R)
    t_inverse = -s_inverse * R_inverse @ t

    inverse_T[:3, :3] = s_inverse * R_inverse
    inverse_T[:3, 3] = t_inverse
    return inverse_T

def apply_transform(points, T):
    homo_points = np.append(points, np.ones((points.shape[0],1)), axis=1)
    after_transform = T @ np.transpose(homo_points)
    return np.transpose(after_transform)[:,:3]


def collect(points_size, points_uniform_ratio, points_sigma, points_padding, bbox_padding, in_dir, mesh_dir, out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    mesh_dir = Path(mesh_dir)

    for env_file in tqdm(glob.glob(str(in_dir / '*.npy'))):
        file_name = os.path.basename(env_file)
        env_name = os.path.splitext(file_name)[0]
        art_name_to_data = np.load(env_file, allow_pickle=True).item()

        if 'Train' in env_name:
            temp_out_dir = out_dir / 'training'
            temp_mesh_dir = mesh_dir / 'training_watertight'
        elif 'Test' in env_name:
            temp_out_dir = out_dir / 'testing'
            temp_mesh_dir = mesh_dir / 'testing_watertight'

        for art_name, art_data in art_name_to_data.items():
            datapath = str(temp_out_dir / (env_name +'_'+art_name + '.npz'))
            pointcloud = np2pcd(art_data['noisy_pointcloud'])
            instance_id = art_data['instance_id']
            art_config = art_data['art_config']
            art_mesh_name = art_data['mesh_name']
            T_obj_nocs = art_data['T_obj_nocs']
            T_nocs_obj = T_inverse_with_scaling(T_obj_nocs)

            art_mesh_obj = trimesh.load(
                            str(temp_mesh_dir / art_mesh_name), 
                            process=False
                            )  # visualmesh_identity_wrt_world

            T_world_obj = np.identity(4)
            T_world_obj[:3, :3] = art_config['scale'] * quat2mat(art_config['rotation'])
            T_world_obj[:3, 3] = art_config['position']
            T_obj_world = T_inverse_with_scaling(T_world_obj)
            T_nocs_world = T_nocs_obj @ T_obj_world

            art_mesh_nocs = deepcopy(art_mesh_obj).apply_transform(T_nocs_obj)

            n_points_uniform = int(points_size * points_uniform_ratio)
            n_points_surface = points_size - n_points_uniform

            boxsize = 1 + points_padding
            points_uniform = np.random.rand(n_points_uniform, 3)
            points_uniform = boxsize * (points_uniform - 0.5)
            points_surface = art_mesh_nocs.sample(n_points_surface)
            points_surface += points_sigma * np.random.randn(n_points_surface, 3)
            points = np.concatenate([points_uniform, points_surface], axis=0)

            occupancies = check_mesh_contains(art_mesh_nocs, points, hash_resolution=1000)
            pc_start = np.array(deepcopy(pointcloud).transform(T_nocs_world).points)
            pc_end = pc_start.copy()
            pc_start_end = pc_start.copy()
            pc_end_start = pc_start.copy()
            pc_seg_start = np.zeros(pc_start.shape[0]) == 1
            pc_seg_end = pc_seg_start.copy()
            start_p_occ = points.copy()
            start_occ_list = np.zeros((2, start_p_occ.shape[0]))
            start_occ_list[1,:] = occupancies.copy()

            np.savez(
                datapath,
                pc_start = pc_start,
                pc_end = pc_end,
                pc_start_end = pc_start_end,
                pc_end_start = pc_end_start,
                pc_seg_start = pc_seg_start,
                pc_seg_end = pc_seg_end,
                state_start = 0,
                state_end = 0,
                screw_axis = np.zeros(3),
                screw_moment = np.zeros(3),
                joint_type = 0,
                joint_index = 0,
                start_p_occ = start_p_occ,
                start_occ_list = start_occ_list,
                end_p_occ = start_p_occ.copy(),
                end_occ_list = start_occ_list.copy(),
                T_world_nocs = T_world_obj @ T_obj_nocs,
                T_nocs_obj = T_nocs_obj
            )



if __name__ == "__main__":
    import argparse
    import logging


    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--mesh_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--points_size', type=int, default=100000)
    parser.add_argument('--points_uniform_ratio', type=float, default=0.6)
    parser.add_argument('--points_sigma', type=float, default=0.01)
    parser.add_argument('--points_padding', type=float, default=0.1)
    parser.add_argument('--bbox_padding', type=float, default=0.0)
    args = parser.parse_args()

    collect(
        args.points_size,
        args.points_uniform_ratio,
        args.points_sigma,
        args.points_padding,
        args.bbox_padding,
        args.in_dir,
        args.mesh_dir,
        args.out_dir
    )