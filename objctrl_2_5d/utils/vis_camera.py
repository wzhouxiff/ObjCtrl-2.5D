import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc

def vis_camera(RT_list, rescale_T=1):
    fig = go.Figure()
    showticklabels = True
    visible = True
    # scene_bounds = 1.5
    scene_bounds = 2.0
    base_radius = 2.5
    zoom_scale = 1.5
    fov_deg = 50.0
    
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1), (3, 4)] 
    
    colors = px.colors.qualitative.Plotly
    
    cone_list = []
    n = len(RT_list)
    color_scale = pc.sample_colorscale("Reds", [i / (len(RT_list) - 1) for i in range(len(RT_list))])
    # color_scale = pc.sample_colorscale("Blues ", [0.3 + 0.7 * i / (len(RT_list) - 1) for i in range(len(RT_list))])
    color_scale = pc.sample_colorscale("Blues", [0.4 + 0.6 * i / (len(RT_list) - 1) for i in range(len(RT_list))])
    # color_scale = pc.sample_colorscale("Cividis", [0.3 + 0.7 * i / (len(RT_list) - 1) for i in range(len(RT_list))])
    # color_scale = pc.sample_colorscale("Viridis", [0.3 + 0.7 * i / (len(RT_list) - 1) for i in range(len(RT_list))])
    


    for i, RT in enumerate(RT_list):
        R = RT[:,:3]
        T = RT[:,-1]/rescale_T
        cone = calc_cam_cone_pts_3d_org(R, T, fov_deg, scale=0.15)
        # cone_list.append((cone, (i*1/n, "green"), f"view_{i}"))
        # color = colors[i % len(colors)]  # 从颜色列表中循环选择颜色
        cone_list.append((cone, color_scale[i], f"view_{i}"))

    
    for (cone, clr, legend) in cone_list:
        for (i, edge) in enumerate(edges):
            (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
            (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
            (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
            fig.add_trace(go.Scatter3d(
                x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                line=dict(color=clr, width=6),
                # line={
                #     'size': 30,
                #     'opacity': 0.8,
                # },
                name=legend, showlegend=(i == 0))) 
    fig.update_layout(
                    height=500,
                    autosize=True,
                    # hovermode=False,
                    margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                    
                    showlegend=True,
                    legend=dict(
                        yanchor='bottom',
                        y=0.01,
                        xanchor='right',
                        x=0.99,
                    ),
                    scene=dict(
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1.0),
                        camera=dict(
                            center=dict(x=0.0, y=0.0, z=0.0),
                            up=dict(x=0.0, y=-1.0, z=0.0),
                            eye=dict(x=scene_bounds/2, y=-scene_bounds/2, z=-scene_bounds/2),
                            ),

                        xaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        ),
                            
                        
                        yaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        ),
                            
                        
                        zaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        )
                    ))
    
    return fig


def calc_cam_cone_pts_3d(R_W2C, T_W2C, fov_deg, scale=1.0, set_canonical=False, first_frame_RT=None):
    fov_rad = np.deg2rad(fov_deg)
    R_W2C_inv = np.linalg.inv(R_W2C)

    # 定义视锥体的长度
    height = scale  # 视锥体的高度
    width = height * np.tan(fov_rad / 2)  # 视锥体在给定FOV下的宽度

    # 计算相机中心位置
    T = np.zeros_like(T_W2C) - T_W2C
    T = np.dot(R_W2C_inv, T)
    cam_x, cam_y, cam_z = T

    # 定义视锥体的四个顶点
    corn1 = np.array([width, width, height])
    corn2 = np.array([-width, width, height])
    corn3 = np.array([-width, -width, height])
    corn4 = np.array([width, -width, height])

    # 将顶点从相机坐标转换到世界坐标
    corners = np.stack([corn1, corn2, corn3, corn4]) - T_W2C
    corners = np.dot(R_W2C_inv, corners.T).T

    # 将视锥体顶点与相机中心坐标组合
    xs = [cam_x] + corners[:, 0].tolist()
    ys = [cam_y] + corners[:, 1].tolist()
    zs = [cam_z] + corners[:, 2].tolist()

    return np.array([xs, ys, zs]).T


def calc_cam_cone_pts_3d_org(R_W2C, T_W2C, fov_deg, scale=0.1, set_canonical=False, first_frame_RT=None):
    fov_rad = np.deg2rad(fov_deg)
    R_W2C_inv = np.linalg.inv(R_W2C)

    # Camera pose center:
    T = np.zeros_like(T_W2C) - T_W2C
    T = np.dot(R_W2C_inv, T)
    cam_x = T[0]
    cam_y = T[1]
    cam_z = T[2]
    if set_canonical:
        T = np.zeros_like(T_W2C)
        T = np.dot(first_frame_RT[:,:3], T) + first_frame_RT[:,-1]
        T = T - T_W2C 
        T = np.dot(R_W2C_inv, T)
        cam_x = T[0]
        cam_y = T[1]
        cam_z = T[2]

    # vertex
    corn1 = np.array([np.tan(fov_rad / 2.0), 0.5*np.tan(fov_rad / 2.0), 1.0]) *scale 
    corn2 = np.array([-np.tan(fov_rad / 2.0), 0.5*np.tan(fov_rad / 2.0), 1.0]) *scale
    corn3 = np.array([0, -0.25*np.tan(fov_rad / 2.0), 1.0]) *scale
    corn4 = np.array([0, -0.5*np.tan(fov_rad / 2.0), 1.0]) *scale

    corn1 = corn1 - T_W2C
    corn2 = corn2 - T_W2C
    corn3 = corn3 - T_W2C
    corn4 = corn4 - T_W2C
    
    corn1 = np.dot(R_W2C_inv, corn1)
    corn2 = np.dot(R_W2C_inv, corn2)
    corn3 = np.dot(R_W2C_inv, corn3) 
    corn4 = np.dot(R_W2C_inv, corn4) 

    # Now attach as offset to actual 3D camera position:
    corn_x1 = corn1[0]
    corn_y1 = corn1[1]
    corn_z1 = corn1[2]
    
    corn_x2 = corn2[0]
    corn_y2 = corn2[1]
    corn_z2 = corn2[2]
    
    corn_x3 = corn3[0]
    corn_y3 = corn3[1]
    corn_z3 = corn3[2]
    
    corn_x4 = corn4[0]
    corn_y4 = corn4[1]
    corn_z4 = corn4[2]
            

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, ]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, ]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, ]

    return np.array([xs, ys, zs]).T

    
def vis_camera_rescale(RTs):
    rescale_T = 1.0
    rescale_T = max(rescale_T, np.max(np.abs(RTs[:, :, -1])) / 1.9)
    fig = vis_camera(RTs, rescale_T=rescale_T)
    # fig.show()
    return fig