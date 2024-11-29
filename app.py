import spaces
import os
import gradio as gr

import torch
from gradio_image_prompter import ImagePrompter
from sam2.sam2_image_predictor import SAM2ImagePredictor
from omegaconf import OmegaConf

from objctrl_2_5d.utils.ui_utils import process_image, get_camera_pose, run_segment, run_depth, get_points, undo_points


from cameractrl.inference import get_pipeline
from objctrl_2_5d.objctrl_2_5d import run
from objctrl_2_5d.utils.examples import examples, sync_points


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
    device = torch.device("cuda")
    print(f"Force device to {device} due to ZeroGPU")
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
        run_segment(segmentor),
        [canvas, original_image, mask_logits],
        [mask, mask_output, masked_original_image, mask_logits]
    )
    
    depth_button.click(
        run_depth(d_model_NK),
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
        [camera_pose, camera_pose_vis]
    )
    
    generated_button.click(
        run(pipeline, device),
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
