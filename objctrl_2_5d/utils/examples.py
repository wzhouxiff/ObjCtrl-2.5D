
import ast
CAMERA_MODE = ["Traj2Cam", "Rotate", "Clockwise", "Translate"]


def sync_points(points_str):
    return ast.literal_eval(points_str)  # convert string to list

examples = [
    
    # Traj2Cam
    [
        {"image": "./assets/images/image_1.png"},  # raw_input
        1.0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[0],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        "[[231, 122], [413, 154]]"
    ],
    
    # Cloud 
    [
        {"image": "./assets/images/image_6.png"},  # raw_input
        0,  # rescale
        4.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        -1,    # Tz
        CAMERA_MODE[3],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [] # points
    ],
    
    [
        {"image": "./assets/images/image_6.png"},  # raw_input
        0,  # rescale
        4.0,   # speed
        60,   # angle
        1,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[3],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [] # points
    ],
    
    [
        {"image": "./assets/images/image_6.png"},  # raw_input
        0,  # rescale
        4.0,   # speed
        60,   # angle
        -1,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[3],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [] # points
    ],
    
    # ACW
    [
        {"image": "./assets/images/00043.png"},  # raw_input
        0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[2],  # camera_option
        "Fixed",  # bg_mode
        True,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [] # points
    ],
    
    
    # Rotation
    [
        {"image": "./assets/images/rose_320x576.png"},  # raw_input
        0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[1],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [] # points
    ],
    [
        {"image": "./assets/images/00051.png"},  # raw_input
        0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[1],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [] # points
    ],
    
    # bg
    # Traj2Cam
    [
        {"image": "./assets/images/00034.png"},  # raw_input
        1.0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[0],  # camera_option
        "Fixed",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [[33, 285], [35, 279], [37, 276], [39, 273], [42, 269], [44, 267], [47, 264], [50, 260], [51, 257], [55, 254], [57, 252], [62, 248], [68, 245], [74, 241]] # points
    ],
    
    # Traj2Cam
    [
        {"image": "./assets/images/00034.png"},  # raw_input
        1.0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[0],  # camera_option
        "Reverse",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [[33, 285], [35, 279], [37, 276], [39, 273], [42, 269], [44, 267], [47, 264], [50, 260], [51, 257], [55, 254], [57, 252], [62, 248], [68, 245], [74, 241]] # points
    ],
    
    # Traj2Cam
    [
        {"image": "./assets/images/00034.png"},  # raw_input
        1.0,  # rescale
        1.0,   # speed
        60,   # angle
        0,    # Tx
        0,    # Ty
        0,    # Tz
        CAMERA_MODE[0],  # camera_option
        "Free",  # bg_mode
        False,  # shared_wapring_latents
        True,  # scale_wise_masks
        0.5,   # ds
        0.5,   # dt
        "42",   # seed
        [[33, 285], [35, 279], [37, 276], [39, 273], [42, 269], [44, 267], [47, 264], [50, 260], [51, 257], [55, 254], [57, 252], [62, 248], [68, 245], [74, 241]] # points
    ],
    
    ]