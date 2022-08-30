import os, sys

from trimesh import Scene

from dataset_generation_second_stage import collect
sys.path.append('../')
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import json
import math
from pathlib import Path
import glob

import torch

from transforms3d.quaternions import quat2mat, mat2quat
import open3d as o3d
import json
import numpy as np
import trimesh
from tqdm import tqdm
from src.utils.misc import sample_point_cloud
from src.utils.chamfer import compute_trimesh_chamfer

import pybullet as p
p.connect(p.DIRECT)

def get_scaling_rotation_translation(T):
    sR = T[:3, :3]
    scale_sq = np.transpose(sR) @ sR
    s = np.sqrt(scale_sq[0,0])
    R = sR / s
    t = T[:3, 3]
    return s, R, t


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

def write_urdf(mesh_filename, collision_filename, robot_name):
    template = '''<?xml version='1.0' encoding='UTF-8'?>
<robot name="robot_name">
    <link name="base">
        <visual name="model">
            <geometry>
                <mesh filename="mesh_filename" />
            </geometry>
        </visual>
        <collision>
            <geometry>
                <mesh filename="collision_filename" />
            </geometry>
        </collision>
    </link>
</robot>'''
    template = template.replace('mesh_filename', mesh_filename)
    template = template.replace('collision_filename', collision_filename)
    template = template.replace('robot_name', robot_name)
    return template

def reconstruct(model, generator, device, in_dir, out_dir, mesh_dir, apply_outlier_removal=False, normalization='Ditto'):
    in_dir = Path(in_dir) # testing dataset to reconstruct
    out_dir = Path(out_dir) 
    mesh_dir = Path(mesh_dir)
    test_files = glob.glob(str(in_dir / ('*.npz')))

    chamfer_dic = dict()

    for f in tqdm(test_files):
        data_input = np.load(f)
        
        
        if apply_outlier_removal:
            pc_start = np2pcd(data_input['pc_start'])
            pc_end = np2pcd(data_input['pc_end'])
            pc_start, _ = pc_start.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
            pc_end, _ = pc_end.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)

            pc_start = np.array(pc_start.points)
            pc_end = np.array(pc_end.points)
        else:
            pc_start = data_input['pc_start']
            pc_end = data_input['pc_end']

        if normalization == 'NOCS':
            T_world_nocs = data_input['T_world_nocs']
            T_nocs_obj = data_input['T_nocs_obj']
            T_obj_nocs = np.linalg.inv(T_nocs_obj)
        elif normalization == 'Ditto':
            #remove the commentted out normalization part in datamodule
            bound_max = np.maximum(pc_start.max(0), pc_end.max(0))
            bound_min = np.minimum(pc_start.min(0), pc_end.min(0))
            norm_center = (bound_max + bound_min) / 2
            norm_scale = (bound_max - bound_min).max() * 1.1
            pc_start = (pc_start - norm_center) / norm_scale
            pc_end = (pc_end - norm_center) / norm_scale
            
            
        pc_start, _ = sample_point_cloud(pc_start, 8192)
        pc_end, _ = sample_point_cloud(pc_end, 8192)
        sample = {
            'pc_start': torch.from_numpy(pc_start).unsqueeze(0).to(device).float(),
            'pc_end': torch.from_numpy(pc_end).unsqueeze(0).to(device).float()
        }

        mesh_dict, mobile_points_all, c, stats_dict = generator.generate_mesh(sample)

        pred_mesh = mesh_dict[0]

        obj_filename = os.path.splitext(os.path.basename(f))[0]
        scene_name = obj_filename.split('v0')[0]+'v0'
        obj_name_in_scene = obj_filename.split('v0')[-1].lstrip('_')
        urdf_path = out_dir / scene_name / obj_name_in_scene / 'fixed_model.urdf'
        alt_urdf_path = out_dir / scene_name / obj_name_in_scene / 'alt_fixed_model.urdf'
        pred_mesh_path = out_dir / scene_name / obj_name_in_scene / 'model.obj'
        collision_mesh_path = out_dir / scene_name / obj_name_in_scene / 'model_vhacd.obj'

        transformation_path = out_dir / scene_name / obj_name_in_scene / 'tran.txt'
        obs_pt_path = out_dir / scene_name / obj_name_in_scene / 'observed_pcd.npy'

        mesh_filename = 'model.obj'
        collision_filename = 'model_vhacd.obj'
        robot_name = scene_name+'_'+obj_name_in_scene
        urdf_content = write_urdf(mesh_filename, collision_filename, robot_name)
        alt_urdf_content = write_urdf(collision_filename, collision_filename, robot_name)

        parent_dir = out_dir / scene_name / obj_name_in_scene
        parent_dir.mkdir(parents=True, exist_ok=True)

        with open(urdf_path, 'w') as urdf:
            urdf.write(urdf_content)

        with open(alt_urdf_path, 'w') as alt_urdf:
            alt_urdf.write(alt_urdf_content)

        with open(transformation_path, 'w') as t:
            if normalization == "NOCS":
                scale, rotation, position = get_scaling_rotation_translation(T_world_nocs)
                quat = mat2quat(rotation)
                
            elif normalization == "Ditto":
                scale = norm_scale
                rotation = np.eye(3)
                position = norm_center
                quat = mat2quat(rotation)
            
            t.write(str(scale)+'\n'+','.join([str(i) for i in position])+'\n'+','.join([str(i) for i in quat])+'\n')

        
        mesh_f_name = os.path.splitext(os.path.basename(f))[0]
        mesh_f_name = '-v0_'.join(mesh_f_name.split('-v0'))
        # gt_mesh_path = mesh_dir / (mesh_f_name+'.off')
        # gt_mesh = trimesh.load(str(gt_mesh_path), process=False)
        # if normalization == 'NOCS':
        #     gt_mesh.apply_transform(T_nocs_obj)
        # elif normalization == 'Ditto':
        #     bbox = gt_mesh.bounding_box.bounds
        #     loc = (bbox[0] + bbox[1]) / 2
        #     scale = (bbox[1] - bbox[0]).max() * 1.1
        #     gt_mesh.apply_translation(-loc)
        #     gt_mesh.apply_scale(1 / scale)
        # chamfer_dist = compute_trimesh_chamfer(gt_mesh=gt_mesh, pred_mesh=pred_mesh, offset=0, scale=1)
        # chamfer_dic[gt_mesh_path] = chamfer_dist
        np.save(obs_pt_path, data_input['pc_start'])
        trimesh.exchange.export.export_mesh(pred_mesh, pred_mesh_path)
        p.vhacd(str(pred_mesh_path),
                str(collision_mesh_path),
                str(out_dir / scene_name / obj_name_in_scene /'vhacd.log'),
                resolution=1000000)
    return chamfer_dic

    

if __name__ == "__main__":
    import argparse
    import logging
    import torch
    from hydra.experimental import initialize, initialize_config_module, initialize_config_dir, compose
    from omegaconf import OmegaConf
    import hydra

    from src.third_party.ConvONets.conv_onet.generation_two_stage import Generator3D

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--mesh_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--remove_outlier', type=bool, default=False)
    parser.add_argument('--norm', type=str, default='Ditto')
    parser.add_argument('--hydra_config_dir', type=str, default='configs/')
    parser.add_argument('--simplify', type=bool, default=False)
    parser.add_argument('--simplify_nfaces', type=int, default=20000)
    args = parser.parse_args()

    with initialize(config_path=args.hydra_config_dir):
        config = compose(
            config_name='config',
            return_hydra_config=True)
    config.datamodule.opt.train.data_dir = '../data/'
    config.datamodule.opt.val.data_dir = '../data/'

    model = hydra.utils.instantiate(config.model)
    ckpt = torch.load(args.ckpt_dir)
    device = torch.device(0)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model = model.eval().to(device)

    if args.simplify == True:

        generator = Generator3D(
            model.model,
            device=device,
            threshold=0.4,
            seg_threshold=0.5,
            input_type='pointcloud',
            refinement_step=0,
            padding=0.1,
            resolution0=32,
            simplify_nfaces=args.simplify_nfaces
        )

    else:
        generator = Generator3D(
            model.model,
            device=device,
            threshold=0.4,
            seg_threshold=0.5,
            input_type='pointcloud',
            refinement_step=0,
            padding=0.1,
            resolution0=32,
        )
    
    chamfer_dic = reconstruct(model, generator, device, args.in_dir, args.out_dir, args.mesh_dir, apply_outlier_removal=args.remove_outlier,normalization=args.norm)

    print(f'Max chamfer dist: {np.max(list(chamfer_dic.values()))}')
    print(f'Min chamfer dist: {np.min(list(chamfer_dic.values()))}')
    print(f'Avg chamfer dist: {np.mean(list(chamfer_dic.values()))}')


    