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

class PointNetPlusPlusAttnFusion(nn.Module):
    def __init__(self, dim=None, c_dim=128, padding=0.1, attn_kwargs=None):
        super().__init__()
        
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