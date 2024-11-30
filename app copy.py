try:
    import spaces
except:
    pass

import os
import gradio as gr

import torch
from gradio_image_prompter import ImagePrompter
from sam2.sam2_image_predictor import SAM2ImagePredictor
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from copy import deepcopy
import cv2

import torch.nn.functional as F
import torchvision
from einops import rearrange
import tempfile

from objctrl_2_5d.utils.ui_utils import process_image, get_camera_pose, get_subject_points, get_points, undo_points, mask_image
from ZoeDepth.zoedepth.utils.misc import colorize

from cameractrl.inference import get_pipeline
from objctrl_2_5d.utils.examples import examples, sync_points

from objctrl_2_5d.utils.objmask_util import RT2Plucker, Unprojected, roll_with_ignore_multidim, dilate_mask_pytorch
from objctrl_2_5d.utils.filter_utils import get_freq_filter, freq_mix_3d


### Title and Description ###
#### Description ####
title = r"""<h1 align="center">ObjCtrl-2.5D: Training-free Object Control with Camera Poses</h1>"""
# subtitle = r"""<h2 align="center">Deployed on SVD Generation</h2>"""
important_link = r"""
<div align='center'>
 <a href='https://wzhouxiff.github.io/projects/MotionCtrl/assets/paper/MotionCtrl.pdf'>[Paper]</a>
&ensp; <a href='https://wzhouxiff.github.io/projects/MotionCtrl/'>[Project Page]</a>
&ensp; <a href='https://github.com/TencentARC/MotionCtrl'>[Code]</a>
</div>
"""

authors = r"""
<div align='center'>
 <a href='https://wzhouxiff.github.io/'>Zhouxia Wang</a>
&ensp; <a href='https://nirvanalan.github.io/'>Yushi Lan</a>
&ensp; <a href='https://shangchenzhou.com/'>Shanchen Zhou</a>
&ensp; <a href='https://www.mmlab-ntu.com/person/ccloy/index.html'>Chen Change Loy</a>
</div>
"""

affiliation = r"""
<div align='center'>
 <a href='https://www.mmlab-ntu.com/'>S-Lab, NTU Singapore</a>
</div>
"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://github.com/TencentARC/MotionCtrl' target='_blank'><b>ObjCtrl-2.5D: Training-free Object Control with Camera Poses</b></a>.<br>
🔥 ObjCtrl2.5D enables object motion control in a I2V generated video via transforming 2D trajectories to 3D using depth, subsequently converting them into camera poses, 
thereby leveraging the exisitng camera motion control module for object motion control without requiring additional training.<br>
"""

article = r"""
If ObjCtrl2.5D is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/MotionCtrl' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/TencentARC%2FMotionCtrl
)](https://github.com/TencentARC/MotionCtrl)

---

📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{wang2024motionctrl,
  title={Motionctrl: A unified and flexible motion controller for video generation},
  author={Wang, Zhouxia and Yuan, Ziyang and Wang, Xintao and Li, Yaowei and Chen, Tianshui and Xia, Menghan and Luo, Ping and Shan, Ying},
  booktitle={ACM SIGGRAPH 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```

📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>zhouzi1212@gmail.com</b>.

"""

# -------------- initialization --------------

CAMERA_MODE = ["Traj2Cam", "Rotate", "Clockwise", "Translate"]

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# segmentation model
segmentor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", cache_dir="ckpt", device=device)

# depth model
d_model_NK = torch.hub.load('./ZoeDepth', 'ZoeD_NK', source='local', pretrained=True).to(device)

# cameractrl model
config = "configs/svd_320_576_cameractrl.yaml"
model_id = "stabilityai/stable-video-diffusion-img2vid"
ckpt = "checkpoints/CameraCtrl_svd.ckpt"
if not os.path.exists(ckpt):
    os.makedirs("checkpoints", exist_ok=True)
    os.system("wget -c https://huggingface.co/hehao13/CameraCtrl_SVD_ckpts/resolve/main/CameraCtrl_svd.ckpt?download=true")
    os.system("mv CameraCtrl_svd.ckpt?download=true checkpoints/CameraCtrl_svd.ckpt")
model_config = OmegaConf.load(config)


pipeline = get_pipeline(model_id, "unet", model_config['down_block_types'], model_config['up_block_types'],
                        model_config['pose_encoder_kwargs'], model_config['attention_processor_kwargs'],
                        ckpt, True, device)

# segmentor = None
# d_model_NK = None
# pipeline = None

### run the demo ##
# @spaces.GPU(duration=5)
def segment(canvas, image, logits):
    if logits is not None:
        logits *=  32.0
    _, points = get_subject_points(canvas)
    image = np.array(image)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        segmentor.set_image(image)
        input_points = []
        input_boxes = []
        for p in points:
            [x1, y1, _, x2, y2, _] = p
            if x2==0 and y2==0:
                input_points.append([x1, y1])
            else:
                input_boxes.append([x1, y1, x2, y2])
        if len(input_points) == 0:
            input_points = None
            input_labels = None
        else:
            input_points = np.array(input_points)
            input_labels = np.ones(len(input_points))
        if len(input_boxes) == 0:
            input_boxes = None
        else:
            input_boxes = np.array(input_boxes)
        masks, _, logits = segmentor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes,
            multimask_output=False,
            return_logits=True,
            mask_input=logits,
        )
        mask = masks > 0
        masked_img = mask_image(image, mask[0], color=[252, 140, 90], alpha=0.9)
        masked_img = Image.fromarray(masked_img)
        
    return mask[0], masked_img, masked_img, logits / 32.0

# @spaces.GPU(duration=5)
def get_depth(image, points):
    
    depth = d_model_NK.infer_pil(image)    
    colored_depth = colorize(depth, cmap='gray_r') # [h, w, 4] 0-255
    
    depth_img = deepcopy(colored_depth[:, :, :3])
    if len(points) > 0:
        for idx, point in enumerate(points):
            if idx % 2 == 0:
                cv2.circle(depth_img, tuple(point), 10, (255, 0, 0), -1)
            else:
                cv2.circle(depth_img, tuple(point), 10, (0, 0, 255), -1)
            if idx > 0:
                cv2.arrowedLine(depth_img, points[idx-1], points[idx], (255, 255, 255), 4, tipLength=0.5)
    
    return depth, depth_img, colored_depth[:, :, :3]


# @spaces.GPU(duration=80)
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

# -------------- UI definition --------------
with gr.Blocks() as demo:
    # layout definition
    gr.Markdown(title)
    gr.Markdown(authors)
    gr.Markdown(affiliation)
    gr.Markdown(important_link)
    gr.Markdown(description)
    
    
    # with gr.Row():
    #     gr.Markdown("""# <center>Repositioning the Subject within Image </center>""")
    mask = gr.State(value=None) # store mask
    removal_mask = gr.State(value=None) # store removal mask
    selected_points = gr.State([]) # store points
    selected_points_text = gr.Textbox(label="Selected Points", visible=False)
    
    original_image = gr.State(value=None) # store original input image
    masked_original_image = gr.State(value=None) # store masked input image
    mask_logits = gr.State(value=None) # store mask logits
    
    depth = gr.State(value=None) # store depth
    org_depth_image = gr.State(value=None) # store original depth image
    
    camera_pose = gr.State(value=None) # store camera pose
    
    with gr.Column():
        
        outlines = """
        <font size="5"><b>There are total 5 steps to complete the task.</b></font>
        - Step 1: Input an image and Crop it to a suitable size;
        - Step 2: Attain the subject mask;
        - Step 3: Get depth and Draw Trajectory;
        - Step 4: Get camera pose from trajectory or customize it;
        - Step 5: Generate the final video.
        """
        
        gr.Markdown(outlines)
        
        
        with gr.Row():
            with gr.Column():
                # Step 1: Input Image
                step1_dec = """
                    <font size="4"><b>Step 1: Input Image</b></font>
                    - Select the region using a <mark>bounding box</mark>, aiming for a ratio close to </mark>320:576</mark> (height:width).
                    - All provided images in `Examples` are in 320 x 576 resolution. Simply press `Process` to proceed.
                    """
                step1 = gr.Markdown(step1_dec)
                raw_input = ImagePrompter(type="pil", label="Raw Image", show_label=True, interactive=True)
                # left_up_point = gr.Textbox(value = "-1 -1", label="Left Up Point", interactive=True)
                process_button = gr.Button("Process")
                
            with gr.Column():
                # Step 2: Get Subject Mask
                step2_dec = """
                    <font size="4"><b>Step 2: Get Subject Mask</b></font>
                    - Use the <mark>bounding boxes</mark> or <mark>paints</mark> to select the subject.
                    - Press `Segment Subject` to get the mask. <mark>Can be refined iteratively by updating points<mark>.
                    """
                step2 = gr.Markdown(step2_dec)
                canvas = ImagePrompter(type="pil", label="Input Image", show_label=True, interactive=True) # for mask painting

                select_button = gr.Button("Segment Subject")
                
        with gr.Row():
            with gr.Column():
                mask_dec = """
                    <font size="4"><b>Mask Result</b></font>
                    - Just for visualization purpose. No need to interact.
                """
                mask_vis = gr.Markdown(mask_dec)
                mask_output = gr.Image(type="pil", label="Mask", show_label=True, interactive=False)
            with gr.Column():
                # Step 3: Get Depth and Draw Trajectory
                step3_dec = """
                    <font size="4"><b>Step 3: Get Depth and Draw Trajectory</b></font>
                    - Press `Get Depth` to get the depth image.
                    - Draw the trajectory by selecting points on the depth image. <mark>No more than 14 points</mark>.
                    - Press `Undo point` to remove all points.
                """
                step3 = gr.Markdown(step3_dec)
                depth_image = gr.Image(type="pil", label="Depth Image", show_label=True, interactive=False)
                with gr.Row():
                    depth_button = gr.Button("Get Depth")
                    undo_button = gr.Button("Undo point")
                    
        with gr.Row():
            with gr.Column():
                # Step 4: Trajectory to Camera Pose or Get Camera Pose
                step4_dec = """
                    <font size="4"><b>Step 4: Get camera pose from trajectory or customize it</b></font>
                    - Option 1: Transform the 2D trajectory to camera poses with depth. <mark>`Rescale` is used for depth alignment. Larger value can speed up the object motion.</mark>
                    - Option 2: Rotate the camera with a specific `Angle`.
                    - Option 3: Rotate the camera clockwise or counterclockwise with a specific `Angle`.
                    - Option 4: Translate the camera with `Tx` (<mark>Pan Left/Right</mark>), `Ty` (<mark>Pan Up/Down</mark>), `Tz` (<mark>Zoom In/Out</mark>) and `Speed`.
                """
                step4 = gr.Markdown(step4_dec)
                camera_pose_vis = gr.Plot(None, label='Camera Pose')
                with gr.Row():
                    with gr.Column():
                        speed = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1.0, label="Speed", interactive=True)
                        rescale = gr.Slider(minimum=0.0, maximum=10, step=0.1, value=1.0, label="Rescale", interactive=True)
                        # traj2pose_button = gr.Button("Option1: Trajectory to Camera Pose")
                        
                        angle = gr.Slider(minimum=-360, maximum=360, step=1, value=60, label="Angle", interactive=True)
                        # rotation_button = gr.Button("Option2: Rotate")
                        # clockwise_button = gr.Button("Option3: Clockwise")
                    with gr.Column():
                        
                        Tx = gr.Slider(minimum=-1, maximum=1, step=1, value=0, label="Tx", interactive=True)
                        Ty = gr.Slider(minimum=-1, maximum=1, step=1, value=0, label="Ty", interactive=True)
                        Tz = gr.Slider(minimum=-1, maximum=1, step=1, value=0, label="Tz", interactive=True)
                        # translation_button = gr.Button("Option4: Translate")
                with gr.Row():
                    camera_option = gr.Radio(choices = CAMERA_MODE, label='Camera Options', value=CAMERA_MODE[0], interactive=True)
                with gr.Row():
                    get_camera_pose_button = gr.Button("Get Camera Pose")
                        
            with gr.Column():
                # Step 5: Get the final generated video
                step5_dec = """
                    <font size="4"><b>Step 5: Get the final generated video</b></font>
                    - 3 modes for background: <mark>Fixed</mark>, <mark>Reverse</mark>, <mark>Free</mark>.
                    - Enable <mark>Scale-wise Masks</mark> for better object control.
                    - Option to enable <mark>Shared Warping Latents</mark> and set <mark>stop frequency</mark> for spatial (`ds`) and temporal (`dt`) dimensions. Larger stop frequency will lead to artifacts.
                """
                step5 = gr.Markdown(step5_dec)
                generated_video = gr.Video(None, label='Generated Video')
                
                with gr.Row():
                    seed = gr.Textbox(value = "42", label="Seed", interactive=True)
                    # num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=25, label="Number of Inference Steps", interactive=True)
                    bg_mode = gr.Radio(choices = ["Fixed", "Reverse", "Free"], label="Background Mode", value="Fixed", interactive=True)
                # swl_mode = gr.Radio(choices = ["Enable SWL", "Disable SWL"], label="Shared Warping Latent", value="Disable SWL", interactive=True)
                scale_wise_masks = gr.Checkbox(label="Enable Scale-wise Masks", interactive=True, value=True)
                with gr.Row():
                    with gr.Column():
                        shared_wapring_latents = gr.Checkbox(label="Enable Shared Warping Latents", interactive=True)
                    with gr.Column():
                        ds = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.5, label="ds", interactive=True)
                        dt = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.5, label="dt", interactive=True)
                
                generated_button = gr.Button("Generate")

                

    # # event definition
    process_button.click(
        fn = process_image,
        inputs = [raw_input],
        outputs = [original_image, canvas]
    )
    
    select_button.click(
        segment,
        [canvas, original_image, mask_logits],
        [mask, mask_output, masked_original_image, mask_logits]
    )
    
    depth_button.click(
        get_depth,
        [original_image, selected_points],
        [depth, depth_image, org_depth_image]
    )
    
    depth_image.select(
        get_points,
        [depth_image, selected_points],
        [depth_image, selected_points],
    )
    undo_button.click(
        undo_points,
        [org_depth_image],
        [depth_image, selected_points]
    )
    
    get_camera_pose_button.click(
        get_camera_pose(CAMERA_MODE),
        [camera_option, selected_points, depth, mask, rescale, angle, Tx, Ty, Tz, speed],
        [camera_pose, camera_pose_vis, rescale]
    )
    
    generated_button.click(
        run_objctrl_2_5d,
        [
         original_image,
         mask,
         depth,
         camera_pose,
         bg_mode,
         shared_wapring_latents,
         scale_wise_masks,
         rescale,
         seed,
         ds,
         dt,
        #  num_inference_steps
         ],
        [generated_video],
    )

    gr.Examples(
        examples=examples,
        inputs=[
            raw_input,
            rescale,
            speed,
            angle,
            Tx,
            Ty,
            Tz,
            camera_option,
            bg_mode,
            shared_wapring_latents,
            scale_wise_masks,
            ds,
            dt,
            seed,
            selected_points_text  # selected_points
        ],
        outputs=[generated_video], 
        examples_per_page=10
    )
    
    selected_points_text.change(
        sync_points,
        inputs=[selected_points_text],
        outputs=[selected_points]
    )


    gr.Markdown(article)


demo.queue().launch(share=True)