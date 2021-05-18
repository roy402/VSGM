import torch
from sys import platform
if platform != "win32":
    from torch_geometric.data import Data
else:
    import open3d as o3d
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import quaternion
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.graph_map.utils_graph_map import *#intrinsic_from_fov, load_extrinsic, load_intrinsic, pixel_coord_np, grid, get_cam_coords
from agents.graph_map.graph_map import BasicMap, test_load_img, test_load_meta_data
'''
Neural-SLAM
'''
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.graph_map.Neural_SLAM.env.habitat.utils import visualizations as vu
from agents.graph_map.Neural_SLAM.env.habitat.utils import pose as pu


# 10x10xcfg.GRAPH_MAP.GRAPH_MAP_SIZE_S
# self.map.activate_nodes = set()
class SlamMAP(BasicMap):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.GPU = cfg.SCENE_GRAPH.GPU
        self.map_size_cm = self.cfg.SLAM_MAP.map_size_cm
        self.EMBED_FEATURE_SIZE = cfg.SCENE_GRAPH.EMBED_FEATURE_SIZE
        self.count_episode = 0
        self.timestep = 0
        self.mapper = self.build_mapper()
        self.reset_map()
        self.figure, self.ax = plt.subplots(
            3, 1, figsize=(4, 6*16/9),
            facecolor="whitesmoke",
            num="Thread 0")
        self.net_map_embedding = self._create_map_embedding_model()
        self.net_map_embedding.to(device)

    def build_mapper(self):
        from agents.graph_map.Neural_SLAM.env.utils.map_builder import MapBuilder
        params = {}
        params['frame_width'] = self.cfg.SLAM_MAP.env_frame_width
        params['frame_height'] = self.cfg.SLAM_MAP.env_frame_height
        params['fov'] = self.cfg.SLAM_MAP.hfov
        params['resolution'] = self.cfg.SLAM_MAP.map_resolution
        params['map_size_cm'] = self.cfg.SLAM_MAP.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.cfg.SLAM_MAP.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.cfg.SLAM_MAP.du_scale
        params['vision_range'] = self.cfg.SLAM_MAP.vision_range
        params['visualize'] = False
        params['obs_threshold'] = self.cfg.SLAM_MAP.obs_threshold
        mapper = MapBuilder(params)
        return mapper

    def _create_map_embedding_model(self):
        from agents.graph_map.Neural_SLAM.model import Global_Policy
        self.global_downscaling = self.cfg.SLAM_MAP.global_downscaling
        map_size = self.cfg.SLAM_MAP.map_size_cm // self.cfg.SLAM_MAP.map_resolution
        full_w, full_h = map_size, map_size
        local_w, local_h = int(full_w / self.cfg.SLAM_MAP.global_downscaling),\
            int(full_h / self.cfg.SLAM_MAP.global_downscaling)
        return Global_Policy((1, local_w, local_h), out_shape=self.EMBED_FEATURE_SIZE)

    def reset_map(self):
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.mapper.reset_map(self.map_size_cm)
        self.map = self.mapper.map
        self.collison_map = np.zeros(self.map.shape[:2])
        self.visited_gt = np.zeros(self.map.shape[:2])
        full_map_size = self.cfg.SLAM_MAP.map_size_cm//self.cfg.SLAM_MAP.map_resolution
        self.explorable_map = np.zeros((full_map_size, full_map_size))
        self.count_episode += 1
        self.timestep = 0
        self.last_sim_location = None
        # # Convert pose to cm and degrees for mapper
        # mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
        #                   self.curr_loc_gt[1]*100.0,
        #                   np.deg2rad(self.curr_loc_gt[2]))

        # # Update ground_truth map and explored area
        # fp_proj, self.map, fp_explored, self.explored_map = \
        #     self.mapper.update_map(depth, mapper_gt_pose)

    def update_map(self, depth_image, agent_meta, sgg_result):
        self.timestep += 1
        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change(agent_meta)
        self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt, (dx_gt, dy_gt, do_gt))
        self.last_loc_gt = np.copy(self.curr_loc_gt)

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        agent_view_degrees = agent_meta["cameraHorizon"]
        self.mapper.agent_view_angle = agent_view_degrees

        '''
        depth process
        '''
        depth_image = depth_image[:, :, 0]
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth_image, mapper_gt_pose)

        # torch.Size([1, 1, 480, 480])
        map_tensor = torch.tensor([[self.map]], dtype=torch.float).to(device=self.device)
        self.map_feature = self.net_map_embedding(map_tensor)

        '''
        visualize
        '''
        # self.visualize_graph_map(depth_image)
        return self.map_feature

    def get_sim_location(self, agent_meta):
        x, z, y = \
            agent_meta['position']['x'], agent_meta['position']['y'], agent_meta['position']['z']
        rotation_x, rotation_y, rotation_z = \
            agent_meta["rotation"]["x"], agent_meta["rotation"]["z"], agent_meta["rotation"]["y"]
        # rotation = np.quaternion(0.999916136264801, 0, 0.0132847428321838, 0)
        quat = Rotation.from_euler('xyz', [rotation_x, rotation_y, rotation_z], degrees=True)
        # import pdb ;pdb.set_trace()
        rotation = np.quaternion(*quat.as_quat())

        axis = quaternion.as_euler_angles(rotation)[0]
        if (axis % (2*np.pi)) < 0.1 or (axis % (2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o

    def get_gt_pose_change(self, agent_meta):
        curr_sim_pose = self.get_sim_location(agent_meta)
        if self.last_sim_location is None:
            self.last_sim_location = curr_sim_pose
        dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

    def _get_short_term_goal(self):
        # # Update collision map
        # if action == 1:
        #     x1, y1, t1 = self.last_loc
        #     x2, y2, t2 = self.curr_loc
        #     if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
        #         self.col_width += 2
        #         self.col_width = min(self.col_width, 9)
        #     else:
        #         self.col_width = 1

        #     dist = pu.get_l2_distance(x1, x2, y1, y2)
        #     if dist < args.collision_threshold: #Collision
        #         length = 2
        #         width = self.col_width
        #         buf = 3
        #         for i in range(length):
        #             for j in range(width):
        #                 wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
        #                                 (j-width//2) * np.sin(np.deg2rad(t1)))
        #                 wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
        #                                 (j-width//2) * np.cos(np.deg2rad(t1)))
        #                 r, c = wy, wx
        #                 r, c = int(r*100/args.map_resolution), int(c*100/args.map_resolution)
        #                 [r, c] = pu.threshold_poses([r, c], self.collison_map.shape)
        #                 self.collison_map[r,c] = 1

        # Get last loc ground truth pose
        last_start_x, last_start_y = self.last_loc_gt[0], self.last_loc_gt[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0/self.cfg.SLAM_MAP.map_resolution),
                      int(c * 100.0/self.cfg.SLAM_MAP.map_resolution)]
        last_start = pu.threshold_poses(last_start, self.visited_gt.shape)

        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt
        r, c = start_y_gt, start_x_gt
        start_gt = [int(r * 100.0/self.cfg.SLAM_MAP.map_resolution),
                    int(c * 100.0/self.cfg.SLAM_MAP.map_resolution)]
        start_gt = pu.threshold_poses(start_gt, self.visited_gt.shape)

        steps = 25
        for i in range(steps):
            x = int(last_start[0] + (start_gt[0] - last_start[0]) * (i+1) / steps)
            y = int(last_start[1] + (start_gt[1] - last_start[1]) * (i+1) / steps)
            self.visited_gt[x, y] = 1

    def visualize_graph_map(self, rgb_img, depth_image):
        self._get_short_term_goal()
        dump_dir = "./slam_dump/"
        ep_dir = '{}/{}/'.format(
                        dump_dir, self.count_episode)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)
        # Get ground truth pose
        start_x_gt, start_y_gt, start_o_gt = self.curr_loc_gt

        goal_coor_didnt_use = (0, 0)
        vis_grid = vu.get_colored_map(
            self.map,
            self.collison_map,
            self.visited_gt,
            self.visited_gt,
            goal_coor_didnt_use,
            self.explored_map,
            self.explorable_map,
            self.map*self.explored_map)
        vis_grid = np.flipud(vis_grid)
        vu.visualize(
            self.figure, self.ax, rgb_img, depth_image, vis_grid[:, :, ::-1],
            (start_x_gt, start_y_gt, start_o_gt),
            (start_x_gt, start_y_gt, start_o_gt),
            dump_dir, self.count_episode, self.timestep,
            visualize=True, print_images=True, vis_style=0)

def main():
    import yaml
    import glob
    import json
    sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
    sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
    from agents.sgg import alfred_data_format
    from config import cfg as _C
    if sys.platform == "win32":
        root = r"D:\cvml_project\projections\inverse_projection\data\d2\trial_T20190909_075955_678702\\"
        root = r"D:\cvml_project\projections\inverse_projection\data\d2\trial_T20190909_100908_040512\\"
        semantic_config_file = r"D:\alfred\alfred\models\config\sgg_without_oracle.yaml"
    else:
        root = r"/home/alfred/data/full_2.1.0/train/pick_and_place_simple-RemoteControl-None-Ottoman-208/trial_T20190909_100908_040512/"
        semantic_config_file = "/home/alfred/models/config/graph_map.yaml"
    def win():
        nonlocal _C
        config = _C
        config.SLAM_MAP.map_resolution = 5
        config.SLAM_MAP.map_size_cm = 2400
        config.SLAM_MAP.map_size_cm = 800
        config.SLAM_MAP.agent_max_z = 200
        config.SLAM_MAP.vision_range = 64
        # config.SLAM_MAP.vision_range = 128
        alfred_dataset = alfred_data_format.AlfredDataset(config)
        grap_map = SlamMAP(
            config,
            )
        traj_data_path = root + "traj_data.json"
        with open(traj_data_path, 'r') as f:
            traj_data = json.load(f)
        frames_depth = test_load_img(os.path.join(root, 'depth_images'), traj_data["images"], None).view(-1, 300, 300, 3)
        frames_rgb = test_load_img(os.path.join(root, 'raw_images'), traj_data["images"], None, type_image=".jpg").view(-1, 300, 300, 3)
        agent_meta_data = test_load_meta_data(root, traj_data["images"])
        for i in range(len(frames_depth)):
            depth_image = frames_depth[i]
            rgb_img = frames_rgb[i]
            agent_meta = agent_meta_data['agent_sgg_meta_data'][i]
            # import pdb; pdb.set_trace()
            target = None
            feature = grap_map.update_map(
                np.array(depth_image),
                agent_meta,
                target)
            grap_map.visualize_graph_map(rgb_img, depth_image)

        grap_map.reset_map()

    def linux():
        import time
        start = time.time()
        nonlocal _C
        config = _C
        config.merge_from_file(semantic_config_file)
        # sgg model
        sys.path.insert(0, os.environ['GRAPH_RCNN_ROOT'])
        from lib.config import cfg
        cfg.merge_from_file("/home/graph-rcnn.pytorch/configs/attribute.yaml")
        config['sgg_cfg'] = cfg
        alfred_dataset = alfred_data_format.AlfredDataset(config)
        grap_map = SlamMAP(
            config,
            )

        traj_data_path = root + "traj_data.json"
        with open(traj_data_path, 'r') as f:
            traj_data = json.load(f)
        frames_depth = test_load_img(os.path.join(root, 'depth_images'), traj_data["images"], None).view(-1, 3, 300, 300)
        frames_rgb = test_load_img(os.path.join(root, 'instance_masks'), traj_data["images"], alfred_dataset.trans_meta_data.transforms).view(-1, 3, 300, 300)
        agent_meta_data = test_load_meta_data(root, traj_data["images"])
        for i in range(len(frames_depth)):
            depth_image = frames_depth[i]
            rgb_image = frames_rgb[i]
            agent_meta = agent_meta_data['agent_sgg_meta_data'][i]
            # import pdb; pdb.set_trace()
            feature = grap_map.update_map(
                np.array(depth_image.view(300, 300, 3)),
                agent_meta,
                )
            grap_map.visualize_graph_map(depth_image)

        grap_map.reset_map()
        # time
        end = time.time()
        print(end - start)

    if sys.platform == "win32":
        win()
    else:
        linux()

if __name__ == '__main__':
    main()