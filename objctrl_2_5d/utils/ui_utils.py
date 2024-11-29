import gradio as gr
from PIL import Image
import numpy as np

from copy import deepcopy
import cv2

from objctrl_2_5d.utils.vis_camera import vis_camera_rescale
from objctrl_2_5d.utils.objmask_util import trajectory_to_camera_poses_v1
from objctrl_2_5d.utils.customized_cam import rotation, clockwise, pan_and_zoom


zc_threshold = 0.2
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

def process_image(raw_image):
    
    
    image, points = raw_image['image'], raw_image['points']
    
    print(points)
    
    try:
        assert(len(points)) == 1, "Please select only one point"
        [x1, y1, _, x2, y2, _] = points[0]
        
        image = image.crop((x1, y1, x2, y2))
        image = image.resize((width, height))
    except:
        image = image.resize((width, height))
    
    return image, gr.update(value={'image': image})

# -------------- general UI functionality --------------

def get_subject_points(canvas):
    return canvas["image"], canvas["points"]


def mask_image(image,
               mask,
               color=[255,0,0],
               alpha=0.5):
    """ Overlay mask on image for visualization purpose.
    Args:
        image (H, W, 3) or (H, W): input image
        mask (H, W): mask to be overlaid
        color: the color of overlaid mask
        alpha: the transparency of the mask
    """
    out = deepcopy(image)
    img = deepcopy(image)
    img[mask == 1] = color
    out = cv2.addWeighted(img, alpha, out, 1-alpha, 0, out)
    return out

def get_points(img,
               sel_pix,
               evt: gr.SelectData):
    
    # collect the selected point
    img = np.array(img)
    img = deepcopy(img)
    sel_pix.append(evt.index)
    # only draw the last two points
    # if len(sel_pix) > 2:
    #     sel_pix = sel_pix[-2:]
    points = []
    for idx, point in enumerate(sel_pix):
        if idx % 2 == 0:
            # draw a red circle at the handle point
            cv2.circle(img, tuple(point), 10, (255, 0, 0), -1)
        else:
            # draw a blue circle at the handle point
            cv2.circle(img, tuple(point), 10, (0, 0, 255), -1)
        points.append(tuple(point))
        # draw an arrow from handle point to target point
        # if len(points) == idx + 1:
        if idx > 0:
            cv2.arrowedLine(img, points[idx-1], points[idx], (255, 255, 255), 4, tipLength=0.5)
            # points = []
            
    return img if isinstance(img, np.ndarray) else np.array(img), sel_pix

# clear all handle/target points
def undo_points(original_image):
    return original_image, []


def interpolate_points(points, num_points):
    x = points[:, 0]
    y = points[:, 1]
    
    # Interpolating the points
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    
    x_new = np.interp(t_new, t, x)
    y_new = np.interp(t_new, t, y)
    
    return np.vstack((x_new, y_new)).T # []

def traj2cam(traj, depth, rescale):
    
    traj = np.array(traj)
    trajectory = interpolate_points(traj, num_frames)
    
    center_h_margin, center_w_margin = center_margin, center_margin
    depth_center = np.mean(depth[height//2-center_h_margin:height//2+center_h_margin, width//2-center_w_margin:width//2+center_w_margin])
    
    if rescale == 0:
        rescale = 1
        
    depth_rescale = round(depth_scale_ * rescale / depth_center, 2)
        
    r_depth = depth * depth_rescale
    
    zc = []
    for i in range(num_frames):
        zc.append(r_depth[int(trajectory[i][1]), int(trajectory[i][0])])
    # print(f'zc: {zc}')
    
    ## norm zc
    zc_norm = np.array(zc)
    zc_grad = zc_norm[1:] - zc_norm[:-1]
    zc_grad = np.abs(zc_grad)
    zc_grad = zc_grad[1:] - zc_grad[:-1]
    zc_grad_std = np.std(zc_grad)

    if zc_grad_std > zc_threshold:
        zc = [zc[0]] * num_frames
        
    # print(f'zc_grad_std: {zc_grad_std}, zc_threshold: {zc_threshold}')
    # print(f'zc: {zc}')

    traj_w2c = trajectory_to_camera_poses_v1(trajectory, intrinsics, num_frames, zc=zc) # numpy: [n_frame, 4, 4]
    RTs = traj_w2c[:, :3]
    fig = vis_camera_rescale(RTs)
    
    return RTs, fig, rescale

def get_rotate_cam(angle, depth):
    # mean_depth = np.mean(depth * mask)
    center_h_margin, center_w_margin = center_margin, center_margin
    depth_center = np.mean(depth[height//2-center_h_margin:height//2+center_h_margin, width//2-center_w_margin:width//2+center_w_margin])
    print(f'rotate depth_center: {depth_center}')
    
    RTs = rotation(num_frames, angle, depth_center, depth_center)
    fig = vis_camera_rescale(RTs)
    
    return RTs, fig

def get_clockwise_cam(angle, depth, mask):
    mask = mask.astype(np.float32) # [0, 1]
    mean_depth = np.mean(depth * mask)
    # center_h_margin, center_w_margin = center_margin, center_margin
    # depth_center = np.mean(depth[height//2-center_h_margin:height//2+center_h_margin, width//2-center_w_margin:width//2+center_w_margin])
    
    RTs = clockwise(angle, num_frames)
    
    RTs[:, -1, -1] = mean_depth
    fig = vis_camera_rescale(RTs)
    
    return RTs, fig

def get_translate_cam(Tx, Ty, Tz, depth, mask, speed):
    mask = mask.astype(np.float32) # [0, 1]
    
    mean_depth = np.mean(depth * mask)
    
    T = np.array([Tx, Ty, Tz])
    T = T.reshape(3, 1)
    T = T[None, ...].repeat(num_frames, axis=0)
    
    RTs = pan_and_zoom(T, speed, n=num_frames)
    RTs[:, -1, -1] += mean_depth
    fig = vis_camera_rescale(RTs)
    
    return RTs, fig

def get_camera_pose(camera_mode):
    def trigger_camera_pose(camera_option, selected_points, depth, mask, rescale, angle, Tx, Ty, Tz, speed):
        if camera_option == camera_mode[0]: # traj2cam
            RTs, fig, rescale = traj2cam(selected_points, depth, rescale)
        elif camera_option == camera_mode[1]: # rotate
            RTs, fig = get_rotate_cam(angle, depth)
            rescale = 0.0
        elif camera_option == camera_mode[2]: # clockwise
            RTs, fig = get_clockwise_cam(angle, depth, mask)
            rescale = 0.0
        elif camera_option == camera_mode[3]: # translate
            RTs, fig = get_translate_cam(Tx, Ty, Tz, depth, mask, speed)
            rescale = 0.0
            
        return RTs, fig, rescale
        
    return trigger_camera_pose
