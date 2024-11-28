import os
from scipy.interpolate import griddata as interp_grid
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image


import torch
from packaging import version as pver

import torch.nn.functional as F
        

def trajectory_to_camera_poses_v1(traj, intrinsics, sample_n_frames, zc = 1.0):
    if not isinstance(zc, list):
        assert isinstance(zc, float) or isinstance(zc, int), 'zc should be a float or int or a list of float or int'
        zc = [zc] * traj.shape[0]
    zc = np.array(zc, dtype=traj.dtype)
    xc = (traj[:, 0] - intrinsics[0, 2]) * zc / intrinsics[0, 0]
    yc = (traj[:, 1] - intrinsics[0, 3]) * zc / intrinsics[0, 1]
    
    first_frame_w2c = np.array([
                                [1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]
                            ], dtype=np.float32)
    
    xw = xc[0]
    yw = yc[0]
    zw = zc[0]
    
    # zw = 0
    # print(f'zw: {zw}')
    Tx = xc - xw
    Ty = yc - yw
    Tz = zc - zw
    
    traj_w2c = [first_frame_w2c]
    for i in range(1, sample_n_frames):
        w2c_mat = np.array([
            [1, 0, 0, Tx[i]],
            [0, 1, 0, Ty[i]],
            [0, 0, 1, Tz[i]],
            [0, 0, 0, 1]
        ], dtype=first_frame_w2c.dtype)
        traj_w2c.append(w2c_mat)
    traj_w2c = np.stack(traj_w2c, axis=0)
    
    
    return traj_w2c # [n_frame, 4, 4]

def Unprojected(image_curr: np.array, 
                depth_curr: np.array,
                RTs: np.array,
                H: int = 320, W: int = 576,
                K: np.array = None,
                dtype: np.dtype = np.float32):
    '''
    image_curr: [H, W, c], float, 0-1
    depth_curr: [H, W], float32, in meters
    RTs: [num_frames, 3, 4], the camera poses, w2c
    '''
    x, y = np.meshgrid(np.arange(W, dtype=dtype), np.arange(H, dtype=dtype), indexing='xy') # pixels

    
    # ceter_depth = np.mean(depth_curr[cam.H//2-10:cam.H//2+10, cam.W//2-10:cam.W//2+10])
    
    RTs = RTs.astype(dtype)
    depth_curr = depth_curr.astype(dtype)
    image_curr = image_curr.reshape(H*W, -1).astype(dtype) # [0, 1]
    
    R0, T0 = RTs[0, :, :3], RTs[0, :, 3:4]
    # new_pts_coord_world2 = image_curr

    pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
    new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)) ## new_pts_coord_world2
    new_pts_colors2 = image_curr ## new_pts_colors2

    pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()


    images = []
    for i in tqdm(range(1, RTs.shape[0])):
        R, T = RTs[i, :, :3], RTs[i, :, 3:4]
        
        ### Transform world to pixel
        pts_coord_cam2 = R.dot(pts_coord_world) + T  ### Same with c2w*world_coord (in homogeneous space)
        pixel_coord_cam2 = np.matmul(K, pts_coord_cam2)   #.reshape(3,H,W).transpose(1,2,0).astype(np.float32)

        valid_idx = np.where(np.logical_and.reduce((pixel_coord_cam2[2]>0, 
                                                    pixel_coord_cam2[0]/pixel_coord_cam2[2]>=0, 
                                                    pixel_coord_cam2[0]/pixel_coord_cam2[2]<=W-1, 
                                                    pixel_coord_cam2[1]/pixel_coord_cam2[2]>=0, 
                                                    pixel_coord_cam2[1]/pixel_coord_cam2[2]<=H-1)))[0]
        
        pixel_coord_cam2 = pixel_coord_cam2[:2, valid_idx]/pixel_coord_cam2[-1:, valid_idx]
        # round_coord_cam2 = np.round(pixel_coord_cam2).astype(np.int32)

        x, y = np.meshgrid(np.arange(W, dtype=dtype), np.arange(H, dtype=dtype), indexing='xy')
        grid = np.stack((x,y), axis=-1).reshape(-1,2)
        image2 = interp_grid(pixel_coord_cam2.transpose(1,0), pts_colors[valid_idx], grid, method='linear', fill_value=0).reshape(H,W,-1)

        images.append(image2)
    
    print(f'Total {len(images)} images, each image shape: {images[0].shape}')
    return images


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

def get_relative_pose(cam_params, zero_t_first_frame):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
    source_cam_c2w = abs_c2ws[0]
    if zero_t_first_frame:
        cam_to_origin = 0
    else:
        cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker, rays_o, rays_d

def RT2Plucker(RT, num_frames, sample_size, fx, fy, cx, cy):
    '''
    RT: [num_frames, 3, 4]
    '''
    cam_params = []
    for i in range(num_frames):
        cam_params.append(Camera([0, fx, fy, cx, cy, 0, 0, RT[i].reshape(-1)]))

    print(cam_params[0].c2w_mat.shape)

    intrinsics = np.asarray([[cam_param.fx * sample_size[1],
                                cam_param.fy * sample_size[0],
                                cam_param.cx * sample_size[1],
                                cam_param.cy * sample_size[0]]
                                for cam_param in cam_params], dtype=np.float32)
    intrinsics = torch.as_tensor(intrinsics)[None]  

    print(intrinsics.shape)

    relative_pose = True
    zero_t_first_frame = True
    use_flip = False

    if relative_pose:
        c2w_poses = get_relative_pose(cam_params, zero_t_first_frame)
    else:
        c2w_poses = np.array([cam_param.c2w_mat for cam_param in cam_params], dtype=np.float32)
    c2w = torch.as_tensor(c2w_poses)[None]                          # [1, n_frame, 4, 4]

    flip_flag = torch.zeros(num_frames, dtype=torch.bool, device=c2w.device)

    plucker_embedding, rays_o, rays_d = ray_condition(intrinsics, c2w, sample_size[0], sample_size[1], device='cpu',
                                        flip_flag=flip_flag)
    plucker_embedding = plucker_embedding[0].permute(0, 3, 1, 2).contiguous() # V, 6, H, W

    plucker_embedding = plucker_embedding.permute(1, 0, 2, 3).contiguous() # 6, V, H, W
    
    return plucker_embedding, rays_o, rays_d


def visualized_trajectories(images, trajectories, save_path, save_each_frame=False):
    '''
    images: [n_frame, H, W, 3], numpy, 0-255
    trajectories: [[n_frame, 2]], list[numpy], x, y
    save_path: str, end with .gif
    '''
    pil_image = []
    H, W = images.shape[1], images.shape[2]
    n_frame = images.shape[0]
    for i in range(n_frame):
        image = images[i].astype(np.uint8)
        image = cv2.UMat(image)
        # print(f'image: {image.shape} {image.dtype} {type(image)}')
        # 
        for traj in trajectories:
            line_data = traj[:i+1]
            if len(line_data) == 1:
                y = int(round(line_data[0][1]))
                x = int(round(line_data[0][0]))
                if y >= H:
                    y = H - 1
                if line_data[0][0] >= W:
                    x = W - 1
                # image[y, x] = [255, 0, 0]
                cv2.circle(image, (x, y), 1, (0, 255, 0), 8)
            else:
                
                for j in range(1, len(line_data)):
                    x0, y0 = int(round(line_data[j-1][0])), int(round(line_data[j-1][1]))
                    x1, y1 = int(round(line_data[j][0])), int(round(line_data[j][1]))
                    if y0 >= H:
                        y0 = H - 1
                    if y1 >= H:
                        y1 = H - 1
                    if x0 >= W:
                        x0 = W - 1
                    if x1 >= W:
                        x1 = W - 1
                    if x0 != x1 or y0 != y1:
                        if j == len(line_data) - 1:
                            line_length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                            arrow_head_length = 10
                            tip_length = arrow_head_length / line_length
                            cv2.arrowedLine(image, (x0, y0), (x1, y1), (255, 0, 0), 6, tipLength=tip_length)
                        else:
                            cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 6)
                cv2.circle(image, (x, y), 1, (0, 255, 0), 8)
                # cv2.circle(image, (x1, y1), 1, (0, 0, 255), 5)
        image = cv2.UMat.get(image)
        pil_image.append(Image.fromarray(image))
        
    pil_image[0].save(save_path, save_all=True, append_images=pil_image[1:], loop=0, duration=100)
    
    # save each frame
    if save_each_frame:
        img_save_root = save_path.replace('.gif', '')
        os.makedirs(img_save_root, exist_ok=True)
        for i, img in enumerate(pil_image):
            img.save(os.path.join(img_save_root, f'{i:05d}.png'))
        
def roll_with_ignore_multidim(arr, shifts):
    result = np.copy(arr)
    for axis, shift in enumerate(shifts):
        result = roll_with_ignore(result, shift, axis)
    return result

def roll_with_ignore(arr, shift, axis):
    result = np.zeros_like(arr)
    if shift > 0:
        result[tuple(slice(shift, None) if i == axis else slice(None) for i in range(arr.ndim))] = \
            arr[tuple(slice(None, -shift) if i == axis else slice(None) for i in range(arr.ndim))]
    elif shift < 0:
        result[tuple(slice(None, shift) if i == axis else slice(None) for i in range(arr.ndim))] = \
            arr[tuple(slice(-shift, None) if i == axis else slice(None) for i in range(arr.ndim))]
    else:
        result = arr
    return result


def dilate_mask_pytorch(mask, kernel_size=2):
    '''
    mask: torch.Tensor, shape [b, c, h, w]
    kernel_size: int
    '''
    
    # Define a smaller kernel for the dilation
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device)
    
    # Perform the dilation operation
    dilated_mask = F.conv2d(mask, kernel, padding=kernel_size//2)
    
    # Ensure the output is still a binary mask (0 and 1)
    dilated_mask = (dilated_mask > 0).to(mask.dtype).to(mask.device)
    
    return dilated_mask

def smooth_mask(mask, kernel_size=3):
    '''
    mask: torch.Tensor, shape [b, c, h, w]
    kernel_size: int
    '''
    
    # Define a Gaussian kernel for smoothing
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=mask.dtype, device=mask.device) / (kernel_size * kernel_size)
    
    # Perform the smoothing operation
    smoothed_mask = F.conv2d(mask, kernel, padding=kernel_size//2)
    
    # Ensure the output is still a binary mask (0 and 1)
    smoothed_mask = (smoothed_mask > 0.5).to(mask.dtype).to(mask.device)
    
    return smoothed_mask
