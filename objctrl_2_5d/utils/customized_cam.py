import numpy as np

def rotation(num_poses, rotation_angle=360, radius=1.0, height=0.81):
    '''
    Input:
        num_poses: number of poses
        rotation_angle: angle of rotation in degrees
        radius: radius of rotation
        height: height of the camera above the ground
        
    Output:
        poses: list of rotation matrices and translation vectors
    '''
    
    poses = []
    
    rotation_angle_rad = np.deg2rad(rotation_angle)
    angle_step = rotation_angle_rad / num_poses

    for i in range(num_poses):
        
        theta = i * angle_step
        
        # Rotation matrix
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
        
        # Translation vector
        t = np.array([0.01 * np.sin(theta), 0, height - 0.01 * np.cos(theta)])
        
        # Combine rotation matrix and translation vector into RT matrix
        RT = np.hstack((R, t.reshape(-1, 1)))
        poses.append(RT)
        
    poses = np.stack(poses, axis=0)
    
    return poses

def clockwise(angle, n_frames):
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Determine the direction of rotation based on the sign of the angle
    if angle_rad < 0:
        clockwise = True
        angle_rad = -angle_rad  # Make the angle positive for calculation
    else:
        clockwise = False
    
    # Generate rotation matrices for each frame
    rotation_matrices = []
    for i in range(n_frames):
        theta = i * angle_rad / (n_frames - 1)
        if clockwise:
            theta = -theta
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        rotation_matrices.append(R)
    
    # Generate translation vectors (assuming no translation)
    translation_vectors = [np.zeros((3, 1)) for _ in range(n_frames)]
    
    # Combine rotation matrices and translation vectors into RT matrices
    RT_matrices = []
    for R, T in zip(rotation_matrices, translation_vectors):
        RT = np.hstack((R, T))
        RT_matrices.append(RT)
        
    RT_matrices = np.stack(RT_matrices, axis=0)
    
    return RT_matrices

def pan_and_zoom(T, speed, base_T=1.5, n=16):
    RT = []
    for i in range(n):
        R = np.array([[1.0, 0.0, 0.0],
                      [0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        _T=(i/n)*speed*base_T*(T[i])
        _RT = np.concatenate([R,_T], axis=1)
        RT.append(_RT)
    RT = np.stack(RT)
    
    return RT