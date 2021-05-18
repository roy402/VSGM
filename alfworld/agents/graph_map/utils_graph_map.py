import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from icecream import ic


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten()), np.ones_like(x.flatten()))).astype(float)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten()))).astype(float)


def bbox_coord_np(xmin, xmax, ymin, ymax):
    # import pdb; pdb.set_trace()
    x = np.linspace(xmin, xmax, abs(xmax-xmin)).astype(np.int)
    y = np.linspace(ymin, ymax, abs(ymax-ymin)).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten()), np.ones_like(x.flatten()))).astype(float)


def load_extrinsic(path_agent):
    eps = 1e-10
    if type(path_agent) == str:
        with open(path_agent, 'rt') as handle:
            a = json.load(handle)
    else:
        a = path_agent
    # extrinsic
    rotation_x, rotation_y, rotation_z = np.deg2rad(0), np.deg2rad(0), 0
    rotation_x, rotation_y, rotation_z = np.deg2rad(
        a["cameraHorizon"]+eps), np.deg2rad(a["rotation"]["y"]+eps), 0
    vector = np.array([a['position']['x'], 0, 0])
    vector = np.array([0, 0, a['position']['x']])
    vector = np.array([a['position']['z'], a['position']['y'], a['position']['x']])
    vector = np.array([a['position']['x'], a['position']['y'], a['position']['z']])
    # ic(rotation_x)
    # ic(rotation_y)
    # ic(vector)
    RX = np.array([[1, 0., 0., 0.],
                   [0., np.cos(rotation_x), -np.sin(rotation_x), 0.],
                   [0., np.sin(rotation_x), np.cos(rotation_x), 0.],
                   [0., 0., 0., 1.]])
    RY = np.array([[np.cos(rotation_y), 0., np.sin(rotation_y), 0.],
                   [0., 1, 0., 0.],
                   [-np.sin(rotation_y), 0., np.cos(rotation_y), 0.],
                   [0., 0., 0., 1.]])
    RZ = np.array([[np.cos(rotation_z), -np.sin(rotation_z), 0., 0.],
                   [np.sin(rotation_z), np.cos(rotation_z), 0., 0.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
    R = RX @ RY @ RZ
    R_inv = np.linalg.inv(R)
    C = np.identity(4)
    C[:3, 3] = vector[:3]
    # C[:3, 3] = -(R @ C)[:3, 3]
    # C[:3, 3] = -(R_inv @ C)[:3, 3]
    # C = -(R_inv @ C)
    # C = -(R @ C)
    e = C @ R
    # import pdb; pdb.set_trace()

    # R_T = np.transpose(R)
    # neg_C = np.identity(4)
    # neg_C[:3, 3] = -vector[:3]
    # e = R_T @ neg_C
    # ic(R)
    # ic(C)
    # ic(e)
    return e, vector


def get_cam_coords(depth, agent, bboxs, labels, K, pixel_coords):
    '''
    IMG
    '''
    b = cv2.split(depth)[0]
    depth = b/255
    # depth = depth.flatten()
    '''
    bboxs pixel coords
    '''
    bboxs = bboxs.detach().cpu().numpy().astype(int)
    labels = labels.detach().cpu().numpy().astype(int)
    pixel_coords = np.array(pixel_coords, copy=True)
    bbox_coords = []
    depth_coords = []
    label_coords = []

    # import pdb; pdb.set_trace()
    for (bbox, label) in zip(bboxs, labels):
        xmin, ymin, xmax, ymax = bbox
        bbox_coord = bbox_coord_np(xmin, xmax, ymin, ymax)
        bbox_coords.append(bbox_coord)
        # depth[y, x]
        depth_coords.append(depth[bbox_coord[1].astype(int), bbox_coord[0].astype(int)])
        label_coords.append([label]*len(bbox_coord[0]))

    # (4, 84612)
    bbox_pixel_coords = np.concatenate(bbox_coords, axis=1)
    # (84612,)
    depth_coords = np.concatenate(depth_coords, axis=0)
    # (84612,)
    label_coords = np.concatenate(label_coords, axis=0)

    '''
    P
    '''
    e, vector = load_extrinsic(agent)
    P = K @ e

    P_inv = np.linalg.inv(P)

    # eps = 1e-10
    # bbox_pixel_coords[3, ] = bbox_pixel_coords[3, ]/(depth_coords + eps)
    # cam_coords = P_inv @ bbox_pixel_coords * depth_coords
    # cam_coords = cam_coords[:3, ]/(cam_coords[3, ] + eps)

    cam_coords = P_inv[:3, :3] @ bbox_pixel_coords[:3, ]
    cam_coords = cam_coords * depth_coords
    # cam_coords = cam_coords-P_inv[:3, 3, None]
    cam_coords = np.vstack([cam_coords, label_coords])
    return cam_coords


def grid(cam_points, V=5, S=100, KEEP_DISPLAY=False):
    x, y, z = cam_points[:3]
    r = V/S
    # import pdb; pdb.set_trace()
    D = np.max(z)
    plt.cla()
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [plt.close() if event.key == 'escape' else None])
    plt.plot(x, z, "xr")
    plt.plot(0.0, 0.0, "ob")
    plt.gca().set_xticks(np.arange(-V/2, V/2, r))
    plt.gca().set_yticks(np.arange(-V/2, V/2, r))
    plt.grid(True)
    if sys.platform == "win32":
        if KEEP_DISPLAY:
            plt.show()
        else:
            plt.pause(1.0)
    else:
        plt.savefig("./grid_mapping.png")


def project_topview(cam_points):
    """
    Draw the topview projection
    """
    max_longitudinal = 70
    max_longitudinal = 500
    window_x = (-max_longitudinal, max_longitudinal)
    window_y = (-max_longitudinal, max_longitudinal)
    # window_x = (-50, 50)
    # window_y = (-3, max_longitudinal)

    x, y, z = cam_points
    print("x max: ", np.max(x))
    print("x min: ", np.min(x))
    print("y max: ", np.max(y))
    print("y min: ", np.min(y))
    print("z max: ", np.max(z))
    print("z min: ", np.min(z))
    # flip the y-axis to positive upwards
    y = - y

    # We sample points for points less than 70m ahead and above ground
    # Camera is mounted 1m above on an ego vehicle
    ind = np.where((z < max_longitudinal) & (y > -1.2))
    ind = np.where(z)
    # ind = np.where((z < max_longitudinal))
    bird_eye = cam_points[:3, ind]
    print(bird_eye.shape)
    print(bird_eye[0:2:2, :].shape)

    # Color by radial distance
    dists = np.sqrt(np.sum(bird_eye[0:2:2, :] ** 2, axis=0))
    axes_limit = 10
    colors = np.minimum(1, dists / axes_limit / np.sqrt(2))

    # Draw Points
    fig, axes = plt.subplots(figsize=(12, 12))
    axes.scatter(bird_eye[0, :], bird_eye[2, :], c=colors, s=0.1)
    axes.set_xlim(window_x)
    axes.set_ylim(window_y)
    axes.set_title('Bird Eye View')
    plt.axis('off')

    plt.gca().set_aspect('equal')
    plt.show()
