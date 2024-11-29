import spaces

import os
import torch
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision
from PIL import Image
from einops import rearrange
import tempfile

from objctrl_2_5d.utils.objmask_util import RT2Plucker, Unprojected, roll_with_ignore_multidim, dilate_mask_pytorch
from objctrl_2_5d.utils.filter_utils import get_freq_filter, freq_mix_3d

DEBUG = False

if DEBUG:
    cur_OUTPUT_PATH = 'outputs/tmp'
    os.makedirs(cur_OUTPUT_PATH, exist_ok=True)

# num_inference_steps=25
min_guidance_scale = 1.0
max_guidance_scale = 3.0

area_ratio = 0.3
depth_scale_ = 5.2
center_margin = 10

height, width = 320, 576
num_frames = 14

intrinsics = np.array([[float(width), float(width), float(width) / 2, float(height) / 2]])
intrinsics = np.repeat(intrinsics, num_frames, axis=0) # [n_frame, 4]
fx = intrinsics[0, 0] / width
fy = intrinsics[0, 1] / height
cx = intrinsics[0, 2] / width
cy = intrinsics[0, 3] / height

down_scale = 8
H, W = height // down_scale, width // down_scale
K = np.array([[width / down_scale, 0, W / 2], [0, width / down_scale, H / 2], [0, 0, 1]])

@spaces.GPU(duration=50)
def run(pipeline, device):
    def run_objctrl_2_5d(condition_image, 
                         mask, 
                         depth, 
                         RTs, 
                         bg_mode, 
                         shared_wapring_latents, 
                         scale_wise_masks, 
                         rescale, 
                         seed, 
                         ds, dt, 
                         num_inference_steps=25):
        
        seed = int(seed)
                
        center_h_margin, center_w_margin = center_margin, center_margin
        depth_center = np.mean(depth[height//2-center_h_margin:height//2+center_h_margin, width//2-center_w_margin:width//2+center_w_margin])
        
        if rescale > 0:
            depth_rescale = round(depth_scale_ * rescale / depth_center, 2)
        else:
            depth_rescale = 1.0
            
        depth = depth * depth_rescale
        
        depth_down = F.interpolate(torch.tensor(depth).unsqueeze(0).unsqueeze(0), 
                                    (H, W), mode='bilinear', align_corners=False).squeeze().numpy() # [H, W]
        
        ## latent
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        latents_org = pipeline.prepare_latents(
                1,
                14,
                8,
                height,
                width,
                pipeline.dtype,
                device,
                generator,
                None,
            )
        latents_org = latents_org / pipeline.scheduler.init_noise_sigma
        
        cur_plucker_embedding, _, _ = RT2Plucker(RTs, RTs.shape[0], (height, width), fx, fy, cx, cy) # 6, V, H, W
        cur_plucker_embedding = cur_plucker_embedding.to(device)
        cur_plucker_embedding = cur_plucker_embedding[None, ...] # b 6 f h w
        cur_plucker_embedding = cur_plucker_embedding.permute(0, 2, 1, 3, 4) # b f 6 h w
        cur_plucker_embedding = cur_plucker_embedding[:, :num_frames, ...]
        cur_pose_features = pipeline.pose_encoder(cur_plucker_embedding)
        
        # bg_mode = ["Fixed", "Reverse", "Free"]
        if bg_mode == "Fixed":
            fix_RTs = np.repeat(RTs[0][None, ...], num_frames, axis=0) # [n_frame, 4, 3]
            fix_plucker_embedding, _, _ = RT2Plucker(fix_RTs, num_frames, (height, width), fx, fy, cx, cy) # 6, V, H, W
            fix_plucker_embedding = fix_plucker_embedding.to(device)
            fix_plucker_embedding = fix_plucker_embedding[None, ...] # b 6 f h w
            fix_plucker_embedding = fix_plucker_embedding.permute(0, 2, 1, 3, 4) # b f 6 h w
            fix_plucker_embedding = fix_plucker_embedding[:, :num_frames, ...]
            fix_pose_features = pipeline.pose_encoder(fix_plucker_embedding)
            
        elif bg_mode == "Reverse":
            bg_plucker_embedding, _, _ = RT2Plucker(RTs[::-1], RTs.shape[0], (height, width), fx, fy, cx, cy) # 6, V, H, W
            bg_plucker_embedding = bg_plucker_embedding.to(device)
            bg_plucker_embedding = bg_plucker_embedding[None, ...] # b 6 f h w
            bg_plucker_embedding = bg_plucker_embedding.permute(0, 2, 1, 3, 4) # b f 6 h w
            bg_plucker_embedding = bg_plucker_embedding[:, :num_frames, ...]
            fix_pose_features = pipeline.pose_encoder(bg_plucker_embedding)
            
        else:
            fix_pose_features = None
            
        #### preparing mask
        
        mask = Image.fromarray(mask)
        mask = mask.resize((W, H))
        mask = np.array(mask).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        
        # visulize mask
        if DEBUG:
            mask_sum_vis = mask[..., 0]
            mask_sum_vis = (mask_sum_vis * 255.0).astype(np.uint8)
            mask_sum_vis = Image.fromarray(mask_sum_vis)
            
            mask_sum_vis.save(f'{cur_OUTPUT_PATH}/org_mask.png')
        
        try:
            warped_masks = Unprojected(mask, depth_down, RTs, H=H, W=W, K=K)
        
            warped_masks.insert(0, mask)
                
        except:
            # mask to bbox
            print(f'!!! Mask is too small to warp; mask to bbox') 
            mask = mask[:, :, 0]
            coords = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(coords)
            # mask[y:y+h, x:x+w] = 1.0
            
            center_x, center_y = x + w // 2, y + h // 2
            center_z = depth_down[center_y, center_x]
            
            # RTs [n_frame, 3, 4] to [n_frame, 4, 4] , add [0, 0, 0, 1]
            RTs = np.concatenate([RTs, np.array([[[0, 0, 0, 1]]] * num_frames)], axis=1)
            
            # RTs: world to camera
            P0 = np.array([center_x, center_y, 1])
            Pc0 = np.linalg.inv(K) @ P0 * center_z
            pw = np.linalg.inv(RTs[0]) @ np.array([Pc0[0], Pc0[1], center_z, 1]) # [4]
            
            P = [np.array([center_x, center_y])]
            for i in range(1, num_frames):
                Pci = RTs[i] @ pw
                Pi = K @ Pci[:3] / Pci[2]
                P.append(Pi[:2])
            
            warped_masks = [mask]
            for i in range(1, num_frames):
                shift_x = int(round(P[i][0] - P[0][0]))
                shift_y = int(round(P[i][1] - P[0][1]))

                cur_mask = roll_with_ignore_multidim(mask, [shift_y, shift_x])
                warped_masks.append(cur_mask)
                
                
            warped_masks = [v[..., None] for v in warped_masks]
                
        warped_masks = np.stack(warped_masks, axis=0) # [f, h, w]
        warped_masks = np.repeat(warped_masks, 3, axis=-1) # [f, h, w, 3]
        
        mask_sum = np.sum(warped_masks, axis=0, keepdims=True)  # [1, H, W, 3]
        mask_sum[mask_sum > 1.0] = 1.0
        mask_sum = mask_sum[0,:,:, 0]
        
        if DEBUG:
            ## visulize warp mask    
            warp_masks_vis = torch.tensor(warped_masks)
            warp_masks_vis = (warp_masks_vis * 255.0).to(torch.uint8)
            torchvision.io.write_video(f'{cur_OUTPUT_PATH}/warped_masks.mp4', warp_masks_vis, fps=10, video_codec='h264', options={'crf': '10'})
            
            # visulize mask
            mask_sum_vis = mask_sum
            mask_sum_vis = (mask_sum_vis * 255.0).astype(np.uint8)
            mask_sum_vis = Image.fromarray(mask_sum_vis)
            
            mask_sum_vis.save(f'{cur_OUTPUT_PATH}/merged_mask.png')
            
        if scale_wise_masks:
            min_area = H * W * area_ratio # cal in downscale
            non_zero_len = mask_sum.sum() 
            
            print(f'non_zero_len: {non_zero_len}, min_area: {min_area}')
            
            if non_zero_len > min_area:
                kernel_sizes = [1, 1, 1, 3]
            elif non_zero_len > min_area * 0.5:
                kernel_sizes = [3, 1, 1, 5]
            else:
                kernel_sizes = [5, 3, 3, 7]
        else:
            kernel_sizes = [1, 1, 1, 1]
            
        mask = torch.from_numpy(mask_sum) # [h, w]
        mask = mask[None, None, ...] # [1, 1, h, w]
        mask = F.interpolate(mask, (height, width), mode='bilinear', align_corners=False) # [1, 1, H, W]
        # mask = mask.repeat(1, num_frames, 1, 1) # [1, f, H, W]
        mask = mask.to(pipeline.dtype).to(device)
        
        ##### Mask End ######
        
        ### Got blending pose features Start ###
    
        pose_features = []
        for i in range(0, len(cur_pose_features)):
            kernel_size = kernel_sizes[i]
            h, w = cur_pose_features[i].shape[-2:]
            
            if fix_pose_features is None:
                pose_features.append(torch.zeros_like(cur_pose_features[i]))
            else:
                pose_features.append(fix_pose_features[i])
                
            cur_mask = F.interpolate(mask, (h, w), mode='bilinear', align_corners=False)
            cur_mask = dilate_mask_pytorch(cur_mask, kernel_size=kernel_size) # [1, 1, H, W]
            cur_mask = cur_mask.repeat(1, num_frames, 1, 1) # [1, f, H, W]
            
            if DEBUG:
                # visulize mask
                mask_vis = cur_mask[0, 0].cpu().numpy() * 255.0
                mask_vis = Image.fromarray(mask_vis.astype(np.uint8))
                mask_vis.save(f'{cur_OUTPUT_PATH}/mask_k{kernel_size}_scale{i}.png')
                
            cur_mask = cur_mask[None, ...] # [1, 1, f, H, W]
            pose_features[-1] = cur_pose_features[i] * cur_mask + pose_features[-1] * (1 - cur_mask)

        ### Got blending pose features End ###
        
        ##### Warp Noise Start ######
        
        if shared_wapring_latents:
            noise = latents_org[0, 0].data.cpu().numpy().copy() #[14, 4, 40, 72]
            noise = np.transpose(noise, (1, 2, 0)) # [40, 72, 4]

            try:
                warp_noise = Unprojected(noise, depth_down, RTs, H=H, W=W, K=K)
                warp_noise.insert(0, noise)
            except:
                print(f'!!! Noise is too small to warp; mask to bbox')
                
                warp_noise = [noise]
                for i in range(1, num_frames):
                    shift_x = int(round(P[i][0] - P[0][0]))
                    shift_y = int(round(P[i][1] - P[0][1]))
                    
                    cur_noise= roll_with_ignore_multidim(noise, [shift_y, shift_x])
                    warp_noise.append(cur_noise)
                    
                warp_noise = np.stack(warp_noise, axis=0) # [f, h, w, 4]
        
            if DEBUG:
                ## visulize warp noise
                warp_noise_vis = torch.tensor(warp_noise)[..., :3] * torch.tensor(warped_masks)
                warp_noise_vis = (warp_noise_vis - warp_noise_vis.min()) / (warp_noise_vis.max() - warp_noise_vis.min())
                warp_noise_vis = (warp_noise_vis * 255.0).to(torch.uint8)
        
                torchvision.io.write_video(f'{cur_OUTPUT_PATH}/warp_noise.mp4', warp_noise_vis, fps=10, video_codec='h264', options={'crf': '10'})
        
        
            warp_latents = torch.tensor(warp_noise).permute(0, 3, 1, 2).to(latents_org.device).to(latents_org.dtype) # [frame, 4, H, W]
            warp_latents = warp_latents.unsqueeze(0) # [1, frame, 4, H, W]
            
            warped_masks = torch.tensor(warped_masks).permute(0, 3, 1, 2).unsqueeze(0) # [1, frame, 3, H, W]
            mask_extend = torch.concat([warped_masks, warped_masks[:,:,0:1]], dim=2) # [1, frame, 4, H, W]
            mask_extend = mask_extend.to(latents_org.device).to(latents_org.dtype)
            
            warp_latents = warp_latents * mask_extend + latents_org * (1 - mask_extend)
            warp_latents = warp_latents.permute(0, 2, 1, 3, 4)
            random_noise = latents_org.clone().permute(0, 2, 1, 3, 4)
                
            filter_shape = warp_latents.shape

            freq_filter = get_freq_filter(
                filter_shape, 
                device = device, 
                filter_type='butterworth',
                n=4,
                d_s=ds,
                d_t=dt
            )
            
            warp_latents = freq_mix_3d(warp_latents, random_noise, freq_filter)
            warp_latents = warp_latents.permute(0, 2, 1, 3, 4)
            
        else:
            warp_latents = latents_org.clone()
            
        generator.manual_seed(42)

        with torch.no_grad():
            result = pipeline(
                image=condition_image,
                pose_embedding=cur_plucker_embedding,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                min_guidance_scale=min_guidance_scale,
                max_guidance_scale=max_guidance_scale,
                do_image_process=True,
                generator=generator,
                output_type='pt',
                pose_features= pose_features,
                latents = warp_latents
            ).frames[0].cpu() #[f, c, h, w]
            
        
        result = rearrange(result, 'f c h w -> f h w c')
        result = (result * 255.0).to(torch.uint8)

        video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
        torchvision.io.write_video(video_path, result, fps=10, video_codec='h264', options={'crf': '8'})
        
        return video_path
    
    return run_objctrl_2_5d
        
    