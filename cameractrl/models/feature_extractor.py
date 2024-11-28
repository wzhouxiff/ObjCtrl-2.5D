# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops import rearrange
from sklearn.cluster import SpectralClustering
from spatracker.blocks import Lie
import matplotlib.pyplot as plt
import cv2

import torch.nn.functional as F
from spatracker.blocks import (
    BasicEncoder,
    CorrBlock,
    EUpdateFormer,
    FusionFormer,
    pix2cam,
    cam2pix,
    edgeMat,
    VitEncoder,
    DPTEnc,
    DPT_DINOv2,
    Dinov2
)

from spatracker.feature_net import (
    LocalSoftSplat
)

from spatracker.model_utils import (
    meshgrid2d, bilinear_sample2d, smart_cat, sample_features5d, vis_PCA
)
from spatracker.embeddings import (
    get_2d_embedding,
    get_3d_embedding,
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed_from_grid,
    Embedder_Fourier,
)
import numpy as np
from spatracker.softsplat import softsplat 

torch.manual_seed(0)


def get_points_on_a_grid(grid_size, interp_shape,
                          grid_center=(0, 0), device="cuda"):
    if grid_size == 1:
        return torch.tensor([interp_shape[1] / 2, 
                             interp_shape[0] / 2], device=device)[
            None, None
        ]

    grid_y, grid_x = meshgrid2d(
        1, grid_size, grid_size, stack=False, norm=False, device=device
    )
    step = interp_shape[1] // 64
    if grid_center[0] != 0 or grid_center[1] != 0:
        grid_y = grid_y - grid_size / 2.0
        grid_x = grid_x - grid_size / 2.0
    grid_y = step + grid_y.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[0] - step * 2
    )
    grid_x = step + grid_x.reshape(1, -1) / float(grid_size - 1) * (
        interp_shape[1] - step * 2
    )

    grid_y = grid_y + grid_center[0]
    grid_x = grid_x + grid_center[1]
    xy = torch.stack([grid_x, grid_y], dim=-1).to(device)
    return xy


def sample_pos_embed(grid_size, embed_dim, coords):
    if coords.shape[-1] == 2:
        pos_embed = get_2d_sincos_pos_embed(embed_dim=embed_dim,
                                             grid_size=grid_size)
        pos_embed = (
            torch.from_numpy(pos_embed)
            .reshape(grid_size[0], grid_size[1], embed_dim)
            .float()
            .unsqueeze(0)
            .to(coords.device)
        )
        sampled_pos_embed = bilinear_sample2d(
            pos_embed.permute(0, 3, 1, 2), 
            coords[:, 0, :, 0], coords[:, 0, :, 1]
        )
    elif coords.shape[-1] == 3:
        sampled_pos_embed = get_3d_sincos_pos_embed_from_grid(
            embed_dim, coords[:, :1, ...]
        ).float()[:,0,...].permute(0, 2, 1)

    return sampled_pos_embed


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        S=8,
        stride=8,
        add_space_attn=True,
        num_heads=8,
        hidden_size=384,
        space_depth=12,
        time_depth=12,
        depth_extend_margin = 0.2,
        args=edict({})
    ):
        super(FeatureExtractor, self).__init__()

        # step1: config the arch of the model
        self.args=args
        # step1.1: config the default value of the model
        if getattr(args, "depth_color", None) == None:
            self.args.depth_color = False
        if getattr(args, "if_ARAP", None) == None:
            self.args.if_ARAP = True
        if getattr(args, "flash_attn", None) == None:
            self.args.flash_attn = True
        if getattr(args, "backbone", None) == None:
            self.args.backbone = "CNN"
        if getattr(args, "Nblock", None) == None:
            self.args.Nblock = 0  
        if getattr(args, "Embed3D", None) == None:
            self.args.Embed3D = True

        # step1.2: config the model parameters
        self.S = S
        self.stride = stride
        self.hidden_dim = 256
        self.latent_dim = latent_dim = 128
        self.b_latent_dim = self.latent_dim//3
        self.corr_levels = 4
        self.corr_radius = 3
        self.add_space_attn = add_space_attn
        self.lie = Lie()
        
        self.depth_extend_margin = depth_extend_margin
        
        

        # step2: config the model components
        # @Encoder
        self.fnet = BasicEncoder(input_dim=3,
            output_dim=self.latent_dim, norm_fn="instance", dropout=0, 
            stride=stride, Embed3D=False
        )

        # conv head for the tri-plane features
        self.headyz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1))
        
        self.headxz = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1))

        # @UpdateFormer
        self.updateformer = EUpdateFormer(
            space_depth=space_depth,
            time_depth=time_depth,
            input_dim=456, 
            hidden_size=hidden_size,
            num_heads=num_heads,
            output_dim=latent_dim + 3,
            mlp_ratio=4.0,
            add_space_attn=add_space_attn,
            flash=getattr(self.args, "flash_attn", True)
        )
        self.support_features = torch.zeros(100, 384).to("cuda") + 0.1

        self.norm = nn.GroupNorm(1, self.latent_dim)
       
        self.ffeat_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.ffeatyz_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )
        self.ffeatxz_updater = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
        )

        #TODO @NeuralArap: optimize the arap
        self.embed_traj = Embedder_Fourier(
            input_dim=5, max_freq_log2=5.0, N_freqs=3, include_input=True
        )
        self.embed3d = Embedder_Fourier(
            input_dim=3, max_freq_log2=10.0, N_freqs=10, include_input=True
        )
        self.embedConv = nn.Conv2d(self.latent_dim+63,
                            self.latent_dim, 3, padding=1)
        
        # @Vis_predictor
        self.vis_predictor = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.embedProj = nn.Linear(63, 456)
        self.zeroMLPflow = nn.Linear(195, 130)

    def prepare_track(self, rgbds, queries):
        """
        NOTE:
        Normalized the rgbs and sorted the queries via their first appeared time
        Args: 
            rgbds: the input rgbd images (B T 4 H W) 
            queries: the input queries (B N 4)
        Return:
            rgbds: the normalized rgbds (B T 4 H W)
            queries: the sorted queries (B N 4)
            track_mask:         
        """
        assert (rgbds.shape[2]==4) and (queries.shape[2]==4)
        #Step1: normalize the rgbs input
        device = rgbds.device
        rgbds[:, :, :3, ...] = 2 * (rgbds[:, :, :3, ...] / 255.0) - 1.0
        B, T, C, H, W = rgbds.shape
        B, N, __ = queries.shape
        self.traj_e = torch.zeros((B, T, N, 3), device=device)
        self.vis_e = torch.zeros((B, T, N), device=device)

        #Step2: sort the points via their first appeared time
        first_positive_inds = queries[0, :, 0].long()
        __, sort_inds = torch.sort(first_positive_inds, dim=0, descending=False)
        inv_sort_inds = torch.argsort(sort_inds, dim=0)
        first_positive_sorted_inds = first_positive_inds[sort_inds]
        # check if can be inverse
        assert torch.allclose(
            first_positive_inds, first_positive_inds[sort_inds][inv_sort_inds]
        )

        # filter those points never appear points during 1 - T
        ind_array = torch.arange(T, device=device)
        ind_array = ind_array[None, :, None].repeat(B, 1, N)
        track_mask = (ind_array >= 
                      first_positive_inds[None, None, :]).unsqueeze(-1)
        
        # scale the coords_init 
        coords_init = queries[:, :, 1:].reshape(B, 1, N, 3).repeat(
            1, self.S, 1, 1
        ) 
        coords_init[..., :2] /= float(self.stride)

        #Step3: initial the regular grid   
        gridx = torch.linspace(0, W//self.stride - 1, W//self.stride)
        gridy = torch.linspace(0, H//self.stride - 1, H//self.stride)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        gridxy = torch.stack([gridx, gridy], dim=-1).to(rgbds.device).permute(
            2, 1, 0
        )
        vis_init = torch.ones((B, self.S, N, 1), device=device).float() * 10

        # Step4: initial traj for neural arap
        T_series = torch.linspace(0, 5, T).reshape(1, T, 1 , 1).cuda() # 1 T 1 1
        T_series = T_series.repeat(B, 1, N, 1)
        # get the 3d traj in the camera coordinates
        intr_init = self.intrs[:,queries[0,:,0].long()]
        Traj_series = pix2cam(queries[:,:,None,1:].double(), intr_init.double()) # [B S N 3]
        #torch.inverse(intr_init.double())@queries[:,:,1:,None].double() # B N 3 1
        Traj_series = Traj_series.repeat(1, 1, T, 1).permute(0, 2, 1, 3).float()
        Traj_series = torch.cat([T_series, Traj_series], dim=-1)
        # get the indicator for the neural arap
        Traj_mask = -1e2*torch.ones_like(T_series)
        Traj_series = torch.cat([Traj_series, Traj_mask], dim=-1)

        return (
            rgbds, 
            first_positive_inds, 
            first_positive_sorted_inds,
            sort_inds, inv_sort_inds, 
            track_mask, gridxy, coords_init[..., sort_inds, :].clone(),
            vis_init, Traj_series[..., sort_inds, :].clone()
            )

    def sample_trifeat(self, t, 
                       coords, 
                       featMapxy,
                       featMapyz,
                       featMapxz):
        """
        Sample the features from the 5D triplane feature map 3*(B S C H W)
        Args:
            t: the time index
            coords: the coordinates of the points B S N 3
            featMapxy: the feature map B S C Hx Wy
            featMapyz: the feature map B S C Hy Wz
            featMapxz: the feature map B S C Hx Wz
        """
        # get xy_t yz_t xz_t
        queried_t = t.reshape(1, 1, -1, 1)
        xy_t = torch.cat(
            [queried_t, coords[..., [0,1]]],
            dim=-1
            )
        yz_t = torch.cat(
            [queried_t, coords[..., [1, 2]]],
            dim=-1
            ) 
        xz_t = torch.cat(
            [queried_t, coords[..., [0, 2]]],
            dim=-1
            )
        featxy_init = sample_features5d(featMapxy, xy_t)
    
        featyz_init = sample_features5d(featMapyz, yz_t)
        featxz_init = sample_features5d(featMapxz, xz_t)
        
        featxy_init = featxy_init.repeat(1, self.S, 1, 1)
        featyz_init = featyz_init.repeat(1, self.S, 1, 1)
        featxz_init = featxz_init.repeat(1, self.S, 1, 1)

        return featxy_init, featyz_init, featxz_init
    
            

    def forward(self, rgbds, queries, num_levels=4, feat_init=None,
                is_train=False, intrs=None, wind_S=None):
        '''
        queries: given trajs (B, f, N, 3) [x, y, z], x, y in camera coordinate, z in depth (need to be normalized)
        vis_init: visibility of the points (B, f, N) , 0 for invisible, 1 for visible
        '''
        B, T, C, H, W = rgbds.shape
        
        Dz = W//self.stride
        
        rgbs_ = rgbds[:, :, :3,...]
        depth_all = rgbds[:, :, 3,...]
        d_near = self.d_near = depth_all[depth_all>0.01].min().item()
        d_far = self.d_far = depth_all[depth_all>0.01].max().item()

        d_near_z = queries.reshape(B, -1, 3)[:, :, 2].min().item()
        d_far_z = queries.reshape(B, -1, 3)[:, :, 2].max().item()
        
        d_near = min(d_near, d_near_z)
        d_far = max(d_far, d_far_z)
        
        d_near = min(d_near - self.depth_extend_margin, 0.01)
        d_far = d_far + self.depth_extend_margin
        
        depths = (depth_all - d_near)/(d_far-d_near)            
        depths_dn = nn.functional.interpolate(
                depths, scale_factor=1.0 / self.stride, mode="nearest")
        depths_dnG = depths_dn*Dz
        
        #Step3: initial the regular grid   
        gridx = torch.linspace(0, W//self.stride - 1, W//self.stride)
        gridy = torch.linspace(0, H//self.stride - 1, H//self.stride)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        gridxy = torch.stack([gridx, gridy], dim=-1).to(rgbds.device).permute(
            2, 1, 0
        ) # 2 H W

        gridxyz = torch.cat([gridxy[None,...].repeat(
                            depths_dn.shape[0],1,1,1), depths_dnG], dim=1)
        Fxy2yz = gridxyz[:,[1, 2], ...] - gridxyz[:,:2]
        Fxy2xz = gridxyz[:,[0, 2], ...] - gridxyz[:,:2]
        if getattr(self.args, "Embed3D", None) == True:
            gridxyz_nm = gridxyz.clone()
            gridxyz_nm[:,0,...] = (gridxyz_nm[:,0,...]-gridxyz_nm[:,0,...].min())/(gridxyz_nm[:,0,...].max()-gridxyz_nm[:,0,...].min())
            gridxyz_nm[:,1,...] = (gridxyz_nm[:,1,...]-gridxyz_nm[:,1,...].min())/(gridxyz_nm[:,1,...].max()-gridxyz_nm[:,1,...].min())
            gridxyz_nm[:,2,...] = (gridxyz_nm[:,2,...]-gridxyz_nm[:,2,...].min())/(gridxyz_nm[:,2,...].max()-gridxyz_nm[:,2,...].min())
            gridxyz_nm = 2*(gridxyz_nm-0.5)
            _,_,h4,w4 = gridxyz_nm.shape
            gridxyz_nm = gridxyz_nm.permute(0,2,3,1).reshape(S*h4*w4, 3)
            featPE = self.embed3d(gridxyz_nm).view(S, h4, w4, -1).permute(0,3,1,2)
            if fmaps_ is None:
                fmaps_ = torch.cat([self.fnet(rgbs_),featPE], dim=1) 
                fmaps_ = self.embedConv(fmaps_)
            else:
                fmaps_new = torch.cat([self.fnet(rgbs_[self.S // 2 :]),featPE[self.S // 2 :]], dim=1) 
                fmaps_new = self.embedConv(fmaps_new)
                fmaps_ = torch.cat(
                    [fmaps_[self.S // 2 :], fmaps_new], dim=0
                )
        else:        
            if fmaps_ is None:
                fmaps_ = self.fnet(rgbs_)
            else:
                fmaps_ = torch.cat(
                [fmaps_[self.S // 2 :], self.fnet(rgbs_[self.S // 2 :])], dim=0
                )

        fmapXY = fmaps_[:, :self.latent_dim].reshape(
            B, T, self.latent_dim, H // self.stride, W // self.stride
        )

        fmapYZ = softsplat(fmapXY[0], Fxy2yz, None,
                        strMode="avg", tenoutH=self.Dz, tenoutW=H//self.stride)
        fmapXZ = softsplat(fmapXY[0], Fxy2xz, None,
                            strMode="avg", tenoutH=self.Dz, tenoutW=W//self.stride)
        
        fmapYZ = self.headyz(fmapYZ)[None, ...]
        fmapXZ = self.headxz(fmapXZ)[None, ...]
        
        # scale the coords_init 
        coords_init = queries[:, :1] # B 1 N 3, the first frame
        coords_init[..., :2] /= float(self.stride)
        
        (featxy_init,
        featyz_init,
        featxz_init) = self.sample_trifeat(
            t=torch.zeros(B*queries.shape[2]),featMapxy=fmapXY,
            featMapyz=fmapYZ,featMapxz=fmapXZ,
            coords = coords_init # B 1 N 3
        )
        
        return torch.stack([featxy_init, featyz_init, featxz_init], dim=-1)

