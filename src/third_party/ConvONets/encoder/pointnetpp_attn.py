"""
From the implementation of https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import torch
import torch.nn as nn

from src.models.modules.Transformer import Attn, TransformerAttn
from src.third_party.ConvONets.encoder.pointnetpp_utils import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
)


# Concat attn with original feature
# return score and abstract point index
# different decoder for occ and articulation

# [modification] Use another class definition
# ---------------------------------------------------------------------------------
'''  
class PointNetPlusPlusAttnFusion(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1_corr = PointNetFeaturePropagation(
            in_channel=256, mlp=[256, 128, c_dim]
        )
        attn_type = attn_kwargs.get("type", "Transformer")
        if attn_type == "simple":
            self.attn = Attn(attn_kwargs)
        elif attn_type == "Transformer":
            self.attn = TransformerAttn(attn_kwargs)

    def encode_deep_feature(self, xyz, return_xyz=False):
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx)
        if return_xyz:
            return l2_points, l2_xyz, fps_idx
        else:
            return l2_points

    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """

        
        xyz = xyz.permute(0, 2, 1)  #32x3xT
        l2_points_xyz2, l2_xyz2, fps_idx2 = self.encode_deep_feature(
            xyz2, return_xyz=True
        )     # pointnetpp encoder 

        l0_points = xyz              # 32x3xT
        l0_xyz = xyz[:, :3, :]       # 32x3xT

        l1_xyz, l1_points, l1_fps_idx = self.sa1(l0_xyz, l0_points, returnfps=True)  # group & sampling + pointnet
        # l1_xyz: 32x3x512, l1_points: 32x128x512, l1_fps_idx: 32x512
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points, returnfps=True)
        # l2_xyz: 32x3x128 l2_points: 32x256x128 l2_fps_idx: 32x128
        fps_idx = torch.gather(l1_fps_idx, 1, l2_fps_idx) # 32x128
        attn, score = self.attn(l2_points, l2_points_xyz2, True)   
        # attention layer: l2_points (before interaction), l2_points_xyz2 (after enteraction)
        # attn: 32x256x128,  score:32x128x128
        l2_points = torch.cat((l2_points, attn), dim=1) # 32x512x128

        l1_points_back = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # 32x256x512
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points_back)       # 32x256xT

        l1_points_corr = self.fp2_corr(l1_xyz, l2_xyz, l1_points, l2_points)   #32x256x512
        l0_points_corr = self.fp1_corr(l0_xyz, l1_xyz, None, l1_points_corr)   #32x256xT

    
        if return_score:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
                fps_idx,
                fps_idx2,
                score,
            )
        else:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points_corr.permute(0, 2, 1),
            )

'''

class PointNetPlusPlusAttnFusion(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()
        '''
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=6,
            mlp=[64, 64, 128],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 + 3,
            mlp=[128, 128, 256],
            group_all=False,
        )

        self.fp2 = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1 = PointNetFeaturePropagation(in_channel=256, mlp=[256, 128, c_dim])

        self.fp2_corr = PointNetFeaturePropagation(in_channel=640, mlp=[512, 256])
        self.fp1_corr = PointNetFeaturePropagation(
            in_channel=256, mlp=[256, 128, c_dim]
        )
        '''
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=6, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128, mlp=[128, 128, c_dim])


    def forward(self, xyz, xyz2, return_score=False):
        """
        xyz: B*N*3
        xyz2: B*N*3
        -------
        return:
        B*N'*3
        B*N'*C
        B*N'
        B*N'
        B*N'*N'
        """

        
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points, fps = self.sa1(l0_xyz, l0_points, returnfps=True)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
    
        if return_score:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                fps,
                fps,
                0,#score,
            )
        else:
            return (
                xyz.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
                l0_points.permute(0, 2, 1),
            )



# ---------------------------------------------------------------------------------