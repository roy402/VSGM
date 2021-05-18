import torch
from sys import platform
if platform != "win32":
    from torch_geometric.data import Data
else:
    import open3d as o3d
import sys
import os
import io
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches
import numpy as np
import json
from icecream import ic
import importlib
from PIL import Image, ImageDraw
import imageio
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.graph_map.utils_graph_map import *#intrinsic_from_fov, load_extrinsic, load_intrinsic, pixel_coord_np, grid, get_cam_coords
from collections import defaultdict, OrderedDict
import glob


class BasicMap(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def reset_map(self):
        raise NotImplementedError

    def update_map(self):
        raise NotImplementedError


class BasicGraphMap(BasicMap):
    def __init__(self, cfg, object_classes_index_to_name):
        super().__init__()
        self.cfg = cfg
        self.object_classes_index_to_name = object_classes_index_to_name
        '''
        Camera Para
        '''
        self.K = intrinsic_from_fov(cfg.GRAPH_MAP.INTRINSIC_HEIGHT, cfg.GRAPH_MAP.INTRINSIC_WIDTH, cfg.GRAPH_MAP.INTRINSIC_FOV)
        self.PIXEL_COORDS = pixel_coord_np(cfg.GRAPH_MAP.INTRINSIC_WIDTH, cfg.GRAPH_MAP.INTRINSIC_HEIGHT)  # [3, npoints]
        '''
        GraphMap Para
        '''
        self.S = cfg.GRAPH_MAP.GRAPH_MAP_SIZE_S
        self.CLASSES = cfg.GRAPH_MAP.GRAPH_MAP_CLASSES
        self.V = cfg.GRAPH_MAP.GRID_COORDS_XY_RANGE_V
        self.R = cfg.GRAPH_MAP.GRID_MIN_SIZE_R
        self.SHIFT_COORDS_HALF_S_TO_MAP = self.S//2
        self.map = np.zeros([self.S, self.S, self.CLASSES]).astype(int)
        self.buffer_plt = []

    def reset_map(self):
        self.map = np.zeros([self.S, self.S, self.CLASSES]).astype(int)
        '''
        visualize
        '''
        with imageio.get_writer('./graph_map_BasicGraphMap_{}_{}_{}.gif'.format(self.S, self.CLASSES, self.R), mode='I', fps=3) as writer:
            for i, buf_file in enumerate(self.buffer_plt):
                pil_img = Image.open(buf_file)
                draw = ImageDraw.Draw(pil_img)
                draw.text((0, 0), str(i), (0, 0, 0))
                plt_img = np.array(pil_img)
                writer.append_data(plt_img)
        self.buffer_plt = []

    def update_map(self, depth_image, agent_meta, sgg_result):
        bboxs = sgg_result["bbox"]
        labels = sgg_result["labels"]
        cam_coords = get_cam_coords(
            depth_image,
            agent_meta,
            bboxs, labels,
            self.K, self.PIXEL_COORDS)
        self.put_label_to_map(cam_coords)
        return cam_coords

    def put_label_to_map(self, cam_coords):
        max_index = self.S-1
        x, y, z, labels = cam_coords
        x = np.round(x / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        z = np.round(z / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        x[x > max_index] = max_index
        z[z > max_index] = max_index
        labels = labels.astype(int)
        self.map[x, z, labels] = labels

    def visualize_graph_map(self, rgb_img, depth_img, KEEP_DISPLAY=False):
        colors = cm.rainbow(np.linspace(0, 1, self.CLASSES))
        Is, Js, Ks = np.where(self.map != 0)
        label_color, legend_color_to_objectId = [], {}
        for i, j, k in zip(Is, Js, Ks):
            label_color.append(colors[self.map[i, j, k]])
            if self.object_classes_index_to_name[self.map[i, j, k]] not in legend_color_to_objectId:
                legend_color_to_objectId[self.object_classes_index_to_name[self.map[i, j, k]]] = \
                    mpatches.Patch(color=colors[self.map[i, j, k]], label=self.object_classes_index_to_name[self.map[i, j, k]])
        # plt.cla()
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [plt.close() if event.key == 'escape' else None])
        # plt.scatter(Is, Js, s=70, c=label_color, cmap="Set2")
        # plt.plot(self.S//2, self.S//2, "ob")
        # plt.gca().set_xticks(np.arange(0, self.S, 1))
        # plt.gca().set_yticks(np.arange(0, self.S, 1))
        # plt.grid(True)
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        # self.buffer_plt.append(buf)
        # # import pdb; pdb.set_trace()
        # if KEEP_DISPLAY:
        #     plt.show()
        # else:
        #     plt.pause(1.0)
        plt.cla()
        plt.clf()
        plt.close()
        plt.gcf().canvas.mpl_connect('key_release_event',
                lambda event: [plt.close() if event.key == 'escape' else None])
        plt.axes(projection='3d').plot3D(self.S//2, self.S//2, self.CLASSES, "ob")
        # plt.axes(projection='3d').scatter3D(Is, Js, Ks, s=70, c=label_color, cmap="Set2")
        plt.axes(projection='3d').scatter3D(Is, Js, Ks, s=70, c=label_color, cmap="Set2")
        plt.gca().set_xticks(np.arange(0, self.S, 1))
        plt.gca().set_yticks(np.arange(0, self.S, 1))
        plt.gca().set_zticks(np.arange(0, self.CLASSES, 1))
        plt.grid(True)
        # legend
        plt.legend(handles=legend_color_to_objectId.values(), scatterpoints=1, loc='lower center', ncol=5, fontsize=8)
        # store
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        self.buffer_plt.append(buf)
        # import pdb; pdb.set_trace()
        if KEEP_DISPLAY:
            plt.show()
        else:
            plt.pause(1.0)
        self.figure, self.ax = plt.subplots(
            3, 1, figsize=(4, 6*16/9),
            facecolor="whitesmoke",
            num="Thread 0")
        ax = self.ax
        ax[0].imshow(rgb_img/255)
        ax[1].imshow(depth_img/255)
        pil_img = Image.open(buf)
        ax[2].imshow(pil_img)

# 10x10xcfg.GRAPH_MAP.GRAPH_MAP_SIZE_S
# self.map.activate_nodes = set()
class GraphMap(BasicGraphMap):
    def __init__(self, cfg, priori_features, dim_rgb_feature, device="cuda", object_classes_index_to_name=None):
        '''
        priori_features: dict. priori_obj_cls_name_to_features, rgb_features, attributes
        '''
        super().__init__(cfg, object_classes_index_to_name)
        '''
        Graph Type
        '''
        self.device = device
        self.priori_features = priori_features
        self.GPU = cfg.SCENE_GRAPH.GPU
        self.dim_rgb_feature = dim_rgb_feature
        self.graphdata_type = getattr(
            importlib.import_module(
                'agents.semantic_graph.semantic_graph'),
            self.cfg.SCENE_GRAPH.GraphData
            )
        self._set_label_to_features()
        self.init_graph_map()

    def _set_label_to_features(self):
        '''
        word & rgb & attributes features
        '''
        features = []
        attributes = []
        # background
        features.append(
            torch.zeros([self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE + self.cfg.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE]))
        attributes.append(
            torch.zeros([self.cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE]))
        # objects
        for k, word_feature in self.priori_features["priori_obj_cls_name_to_features"].items():
            rgb_feature = torch.tensor(self.priori_features["rgb_features"][str(k)]).float()
            feature = torch.cat([word_feature, rgb_feature])
            # [0] is _append_unique_obj_index_to_attribute
            attribute = torch.tensor(self.priori_features["attributes"][str(k)] + [0]).float()
            features.append(feature)
            attributes.append(attribute)
        self.label_to_features = torch.stack(features).to(device=self.device, dtype=torch.float)
        self.label_to_attributes = torch.stack(attributes).to(device=self.device, dtype=torch.float)
        assert len(self.label_to_features) == len(self.priori_features["rgb_features"].keys()), "len diff error"
        assert self.label_to_attributes.shape[-1] == self.cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE, "len diff error"

    def init_graph_map(self):
        self.map = self.graphdata_type(
            self.priori_features["priori_obj_cls_name_to_features"],
            self.GPU,
            self.dim_rgb_feature,
            device=self.device
            )
        '''
        Create graph map node space
        '''
        feature_size = self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE + self.cfg.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE
        attribute_size = self.cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE
        self.map.x = torch.zeros([self.S * self.S * self.CLASSES, feature_size], device=self.device, dtype=torch.float)
        self.map.attributes = torch.zeros([self.S * self.S * self.CLASSES, attribute_size], device=self.device, dtype=torch.float)
        for x in range(self.S):
            for z in range(self.S):
                for label in range(self.CLASSES):
                    target_node_index = x + self.S * z + self.S * self.S * label
                    if label < len(self.label_to_features):
                        self.map.x[target_node_index] = self.label_to_features[label]
                        self.map.attributes[target_node_index] = self.label_to_attributes[label]
                    else:
                        print("label out of bounds: {}. Carefully if sgg predict label out of bounds also".format(label))
                        self.map.x[target_node_index] = self.label_to_features[0]
                        self.map.attributes[target_node_index] = self.label_to_attributes[0]
        self.map.activate_nodes = set(range(self.S * self.S))
        self.map.queue_grid_layer = [0] * self.S * self.S
        '''
        graph map node relation
        '''
        edges = []
        '''
        most top grid connect together
        would be square grid
        '''
        for x in range(self.S):
            for z in range(self.S):
                if x < self.S-1 and z < self.S-1:
                    top_map = x + self.S * z
                    # right edge a -> b
                    edge = torch.tensor(
                        [top_map, top_map+1],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
                    # left edge b -> a
                    edge = torch.tensor(
                        [top_map+1, top_map],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
                    # down edge a -> c
                    edge = torch.tensor(
                        [top_map, top_map+self.S * (z+1)],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
                    # up edge c -> a
                    edge = torch.tensor(
                        [top_map+self.S * (z+1), top_map],
                        device=self.device,
                        dtype=torch.long).contiguous()
                    edges.append(edge)
        '''
        layer node connect to top grid node
        '''
        for x in range(self.S):
            for z in range(self.S):
                '''
                layer ? (src) to top layer (dst)
                '''
                top_map = x + self.S * z
                for label in range(1, self.CLASSES):
                    src = top_map + self.S * self.S * label
                    dst = top_map
                    edge = torch.tensor([src, dst], device=self.device, dtype=torch.long).contiguous()
                    edges.append(edge)
        self.map.edge_obj_to_obj = torch.stack(edges).reshape(2, -1)

    def reset_map(self):
        self.map.activate_nodes = set(range(self.S * self.S))
        self.map.queue_grid_layer = [0] * self.S * self.S
        '''
        visualize
        '''
        with imageio.get_writer('./graph_map_GraphMap_{}_{}_{}.gif'.format(self.S, self.CLASSES, self.R), mode='I', fps=3) as writer:
            for i, buf_file in enumerate(self.buffer_plt):
                pil_img = Image.open(buf_file)
                draw = ImageDraw.Draw(pil_img)
                draw.text((0, 0), str(i), (0, 0, 0))
                plt_img = np.array(pil_img)
                writer.append_data(plt_img)
        self.buffer_plt = []

    def put_label_to_map(self, cam_coords):
        max_index = self.S-1
        x, y, z, labels = cam_coords
        x = np.round(x / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        x[x > max_index] = max_index
        x[x < -max_index] = -max_index
        z = np.round(z / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        z[z > max_index] = max_index
        z[z < -max_index] = -max_index
        # labels.shape (163053,) # array([27, 27, 27, ..., 74, 74, 74])
        labels = labels.astype(int)
        coors = x + self.S * z
        one_dim_coors = self.grid_layer_to_one_dim_coors(coors)
        # len(node_indexs) 163053 -> array([ 27, 127, 227, ..., 937,  37, 137])
        node_indexs = coors + one_dim_coors
        self.map.activate_nodes.update(node_indexs)
        # self.map.x.shape -> torch.Size([1000, 2348])
        self.map.x[node_indexs] = self.label_to_features[labels]
        self.map.attributes[node_indexs] = self.label_to_attributes[labels]

    def grid_layer_to_one_dim_coors(self, coors):
        '''
        grid layer from 0 ~ cfg.GRAPH_MAP.GRAPH_MAP_CLASSES

        coors: [0, 1, 0, 3, 5, 6, 10, 20, 10, 30, 2, ...]. will be < self.S * self.S
        coors = x + self.S * z
        len(self.map.queue_grid_layer) = [0] * self.S * self.S (when grid layer=0)
        '''
        three_dim_to_one = self.S * self.S
        one_dim_coors = []
        for coor in coors:
            # [0~self.CLASSES)
            grid_layer = self.map.queue_grid_layer[coor]
            grid_layer_to_one_dim = three_dim_to_one * grid_layer
            one_dim_coors.append(grid_layer_to_one_dim)
            self.map.queue_grid_layer[coor] = (self.map.queue_grid_layer[coor] + 1) % self.CLASSES
        # queue_grid_layer[:5] [0, 1, 2, 0, 1]
        # coors[:5] array([27, 27, 27, 37, 37])
        return one_dim_coors
        '''
        # Another method get one_dim_coors
        increase_when_same_coors_occur = defaultdict(int)
        # accumulate same coor to increase
        # [0, 1, 0, 3, 5, 6, 10, 20, 10, 30, 2, 0, ...] coors
        # ->
        # [0, 0, 1, 0, 0, 0,  0,  0,  1,  0, 0, 2, ....] each_coor_layer
        each_coor_layer = [0]*len(coors)
        for i in range(len(coors)):
            each_coor_layer[i] = increase_when_same_coors_occur[coors[i]]
            increase_when_same_coors_occur[coors[i]] = (increase_when_same_coors_occur[coors[i]] + 1) % self.CLASSES
        grid_layer = (self.map.queue_grid_layer[coors] + each_coor_layer) % self.CLASSES
        self.map.queue_grid_layer[coors] = grid_layer
        return grid_layer
        '''

    def visualize_graph_map(self):
        print("Didn't implement visualize_graph_mapis")


# self.map.activate_nodes = dict()
class GraphMapV2(GraphMap):
    def __init__(self, cfg, priori_features, dim_rgb_feature, device="cuda", object_classes_index_to_name=None):
        '''
        priori_features: dict. priori_obj_cls_name_to_features, rgb_features, attributes
        '''
        super().__init__(cfg, priori_features, dim_rgb_feature, device, object_classes_index_to_name)

    def init_graph_map(self):
        super().init_graph_map()
        self.map.activate_nodes = dict()
        for i in range(self.S * self.S):
            self.map.activate_nodes[i] = 0

    def reset_map(self):
        super().reset_map()
        self.map.activate_nodes = dict()
        for i in range(self.S * self.S):
            self.map.activate_nodes[i] = 0

    def put_label_to_map(self, cam_coords):
        max_index = self.S-1
        x, y, z, labels = cam_coords
        x = np.round(x / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        x[x > max_index] = max_index
        x[x < -max_index] = -max_index
        z = np.round(z / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        z[z > max_index] = max_index
        z[z < -max_index] = -max_index
        # labels.shape (163053,) # array([27, 27, 27, ..., 74, 74, 74])
        labels = labels.astype(int)
        coors = x + self.S * z
        one_dim_coors = self.grid_layer_to_one_dim_coors(coors)
        # len(node_indexs) 163053 -> array([ 27, 127, 227, ..., 937,  37, 137])
        node_indexs = coors + one_dim_coors
        node_indexs, indices = np.unique(node_indexs, return_index=True)
        labels = labels[indices]
        for node_index, label in zip(node_indexs, labels):
            self.map.activate_nodes[node_index] = label
        # self.map.x.shape -> torch.Size([1000, 2348])
        self.map.x[node_indexs] = self.label_to_features[labels]
        self.map.attributes[node_indexs] = self.label_to_attributes[labels]

        # self.visualize_graph_map()

    def visualize_graph_map(self, rgb_img, depth_img, THREE_DIM_DISPLAY=True):
        CLASSES = 108
        colors = cm.rainbow(np.linspace(0, 1, CLASSES))
        label_color, legend_color_to_objectId = [], {}
        Is, Js, Ks = [], [], []
        two_dim_size = self.S * self.S
        for one_dim_coor, label in self.map.activate_nodes.items():
            # if label == 0:
            #     continue
            # one_dim_coor = x + self.S * z + self.S * self.S * labels
            k = one_dim_coor // two_dim_size
            two_dim_coor = one_dim_coor % two_dim_size
            i, j = two_dim_coor % self.S, two_dim_coor // self.S
            label_color.append(colors[label])
            Is.append(i)
            Js.append(j)
            Ks.append(k)
            # https://moonbooks.org/Articles/How-to-add-a-legend-for-a-scatter-plot-in-matplotlib-/
            if self.object_classes_index_to_name[label] not in legend_color_to_objectId:
                objectId = self.object_classes_index_to_name[label]
                # UserWarning: The handle <matplotlib.patches.Patch object at 0x7fe96198bc18> has a label of '__background__' which cannot be automatically added to the legend.  plt.legend(handles=legend_color_to_objectId.values(), scatterpoints=1, loc='lower center', ncol=5, fontsize=8)
                if objectId == '__background__':
                    objectId = 'background'
                legend_color_to_objectId[self.object_classes_index_to_name[label]] = \
                    mpatches.Patch(color=colors[label], label=objectId)
        plt.cla()
        plt.clf()
        plt.close()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [plt.close() if event.key == 'escape' else None])
        if THREE_DIM_DISPLAY:
            plt.axes(projection='3d').plot3D(self.S//2, self.S//2, self.CLASSES, "ob")
            plt.axes(projection='3d').scatter3D(Is, Js, Ks, s=70, c=label_color, cmap="Set2")
        else:
            plt.plot(self.S//2, self.S//2, "ob")
            plt.scatter(Is, Js, s=70, c=label_color, cmap="Set2")
        plt.gca().set_xticks(np.arange(0, self.S, 1))
        plt.gca().set_yticks(np.arange(0, self.S, 1))
        if THREE_DIM_DISPLAY:
            plt.gca().set_zticks(np.arange(0, self.CLASSES, 1))
        plt.grid(True)
        # legend
        plt.legend(handles=legend_color_to_objectId.values(), scatterpoints=1, loc='lower center', ncol=5, fontsize=8)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        self.buffer_plt.append(buf)

        self.figure, self.ax = plt.subplots(
            3, 1, figsize=(4, 6*16/9),
            facecolor="whitesmoke",
            num="Thread 0")
        ax = self.ax
        # for i in range(3):
        #     ax[i].clear()
        #     ax[i].set_yticks([])
        #     ax[i].set_xticks([])
        #     ax[i].set_yticklabels([])
        #     ax[i].set_xticklabels([])
        ax[0].imshow(rgb_img/255)
        ax[1].imshow(depth_img/255)
        pil_img = Image.open(buf)
        ax[2].imshow(pil_img)

# 10x10x108 (SGG predict label size)
class GraphMap_SXSXLABLE(GraphMap):
    def __init__(self, cfg, priori_features, dim_rgb_feature, device="cuda", object_classes_index_to_name=None):
        '''
        priori_features: dict. priori_obj_cls_name_to_features, rgb_features, attributes
        '''
        super().__init__(cfg, priori_features, dim_rgb_feature, device, object_classes_index_to_name)
        '''
        CLASSES
        '''
        self.CLASSES = len(self.label_to_features)
        print("self.CLASSES = cfg.GRAPH_MAP.GRAPH_MAP_CLASSES would not be use")

        self.init_graph_map()

    def put_label_to_map(self, cam_coords):
        max_index = self.S-1
        x, y, z, labels = cam_coords
        x = np.round(x / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        x[x > max_index] = max_index
        z = np.round(z / self.R).astype(int) + self.SHIFT_COORDS_HALF_S_TO_MAP
        z[z > max_index] = max_index
        labels = labels.astype(int)
        coors = x + self.S * z + self.S * self.S * labels
        activate_node = np.unique(coors).tolist()
        self.map.activate_nodes.extend(activate_node)
        self.map.activate_nodes = list(set(self.map.activate_nodes))
        # ic(self.map.activate_nodes)
        # ic(len(self.map.activate_nodes))


'''
test
'''
def test_load_img(path, list_img_traj, transforms, type_image=".png"):
    def _load_with_path():
        frames_depth = None
        low_idx = -1
        for i, dict_frame in enumerate(list_img_traj):
            # 60 actions need 61 frames
            if low_idx != dict_frame["low_idx"]:
                low_idx = dict_frame["low_idx"]
            else:
                continue
            name_frame = dict_frame["image_name"].split(".")[0]
            frame_path = os.path.join(path, name_frame + type_image)
            if os.path.isfile(frame_path):
                img_depth = Image.open(frame_path).convert("RGB")
                if transforms is not None:
                    img_depth = \
                        transforms(img_depth, None)[0]
                else:
                    img_depth = torch.tensor(np.array(img_depth), dtype=torch.float)
            else:
                print("file is not exist: {}".format(frame_path))
            img_depth = img_depth.unsqueeze(0)

            if frames_depth is None:
                frames_depth = img_depth
            else:
                frames_depth = torch.cat([frames_depth, img_depth], dim=0)
        frames_depth = torch.cat([frames_depth, img_depth], dim=0)
        return frames_depth
    frames_depth = _load_with_path()
    return frames_depth


def test_load_meta_data(root, list_img_traj):
    def agent_sequences_to_one(META_DATA_FILE="meta_agent.json",
                               SGG_META="agent_meta",
                               EXPLORATION_META="exploration_agent_meta",
                               len_meta_data=-1):
        # load
        # print("_load with path", root)
        all_meta_data = {
            "agent_sgg_meta_data": [],
            "exploration_agent_sgg_meta_data": [],
        }
        low_idx = -1
        for i, dict_frame in enumerate(list_img_traj):
            # 60 actions need 61 frames
            if low_idx != dict_frame["low_idx"]:
                low_idx = dict_frame["low_idx"]
            else:
                continue
            name_frame = dict_frame["image_name"].split(".")[0]
            file_path = os.path.join(root, SGG_META, name_frame + ".json")
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    meta_data = json.load(f)
                all_meta_data["agent_sgg_meta_data"].append(meta_data)
            else:
                print("file is not exist: {}".format(file_path))
        all_meta_data["agent_sgg_meta_data"].append(meta_data)
        if len_meta_data != -1:
            n_meta_gap = len(all_meta_data["agent_sgg_meta_data"])-len_meta_data
            for _ in range(n_meta_gap):
                if _ == 0:
                    print("{}. gap num {}".format(root, n_meta_gap))
                    print("meta len should be ", len_meta_data)
                all_meta_data["agent_sgg_meta_data"].append(meta_data)
        exploration_path = os.path.join(root, EXPLORATION_META, "*.json")
        exploration_file_paths = glob.glob(exploration_path)
        for exploration_file_path in exploration_file_paths:
            with open(exploration_file_path, 'r') as f:
                meta_data = json.load(f)
            all_meta_data["exploration_agent_sgg_meta_data"].append(meta_data)
        return all_meta_data

    agent_meta_data = agent_sequences_to_one(
        META_DATA_FILE="meta_agent.json",
        SGG_META="agent_meta",
        EXPLORATION_META="exploration_agent_meta")
    return agent_meta_data


def main():
    import yaml
    import glob
    import json
    sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
    sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
    from agents.sgg import alfred_data_format
    from config import cfg as _C
    if sys.platform == "win32":
        root = r"D:\cvml_project\projections\inverse_projection\data\d2\trial_T20190909_100908_040512\\"
        semantic_config_file = r"D:\alfred\alfred\models\config\sgg_without_oracle.yaml"
    else:
        root = r"/home/alfred/data/full_2.1.0/train/pick_two_obj_and_place-ToiletPaper-None-Drawer-423/trial_T20190907_111226_698552/"
        root = r"/home/alfred/data/full_2.1.0/train/pick_clean_then_place_in_recep-Kettle-None-Cabinet-5/trial_T20190909_043020_330212/"
        root = r"/home/alfred/data/full_2.1.0/train/pick_and_place_simple-RemoteControl-None-Ottoman-208/trial_T20190909_100908_040512/"
        semantic_config_file = "/home/alfred/models/config/graph_map.yaml"
    def win():
        nonlocal _C
        config = _C
        alfred_dataset = alfred_data_format.AlfredDataset(config)
        grap_map = BasicGraphMap(
            config,
            alfred_dataset.trans_meta_data.SGG_result_ind_to_classes,
            )
        traj_data_path = root + "traj_data.json"
        with open(traj_data_path, 'r') as f:
            traj_data = json.load(f)
        frames_depth = test_load_img(os.path.join(root, 'depth_images'), traj_data["images"], None).view(-1, 300, 300, 3)
        agent_meta_data = test_load_meta_data(root, traj_data["images"])
        cat_cam_coords = np.array([[], [], [], []])
        for i in range(10):
            depth_image = frames_depth[i]
            agent_meta = agent_meta_data['agent_sgg_meta_data'][i]
            img, target, idx, rgb_img = alfred_dataset[i]
            bbox = target.bbox
            bbox[bbox>=300] = 299
            # import pdb; pdb.set_trace()
            target = {
                "bbox": bbox,
                "labels": target.get_field("labels"),
            }
            cam_coords = grap_map.update_map(
                np.array(depth_image),
                agent_meta,
                target)
            grap_map.visualize_graph_map()
            cat_cam_coords = np.concatenate([cat_cam_coords, cam_coords], axis=1)

        grap_map.visualize_graph_map(KEEP_DISPLAY=True)
        grap_map.reset_map()

        # Visualize
        pcd_cam = o3d.geometry.PointCloud()
        pcd_cam.points = o3d.utility.Vector3dVector(cat_cam_coords.T[:, :3])
        # Flip it, otherwise the pointcloud will be upside down
        pcd_cam.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # o3d.visualization.draw_geometries([pcd_cam])

        # Do top view projection
        # project_topview(cam_coords)
        # grid(cat_cam_coords, KEEP_DISPLAY=True)
    def linux():
        import time
        start = time.time()
        from agents.sgg import sgg
        from agents.semantic_graph.semantic_graph import SceneGraph
        nonlocal _C
        config = _C
        config.merge_from_file(semantic_config_file)
        config.GRAPH_MAP.GRAPH_MAP_SIZE_S = 20
        config.GRAPH_MAP.GRID_MIN_SIZE_R = 0.1
        config.GRAPH_MAP.GRAPH_MAP_CLASSES = 108
        # sgg model
        sys.path.insert(0, os.environ['GRAPH_RCNN_ROOT'])
        from lib.config import cfg
        cfg.merge_from_file("/home/graph-rcnn.pytorch/configs/attribute.yaml")
        config['sgg_cfg'] = cfg
        alfred_dataset = alfred_data_format.AlfredDataset(config)
        cfg.MODEL.SAVE_SGG_RESULT = True
        sgg_model = sgg.load_pretrained_model(
            cfg,
            alfred_dataset.trans_meta_data.transforms,
            alfred_dataset.trans_meta_data.SGG_result_ind_to_classes,
            "cuda:%d" % config.SGG.GPU,
            )
        # init class
        scene_graph = SceneGraph(
            config,
            alfred_dataset.trans_meta_data.SGG_result_ind_to_classes,
            config.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE,
            "cuda",
            )
        # GraphMap(  GraphMapV2(
        grap_map = GraphMapV2(
            config,
            scene_graph.priori_features,
            config.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE,
            "cuda",
            object_classes_index_to_name=scene_graph.object_classes_index_to_name,
            )
        grap_map = BasicGraphMap(
            config,
            alfred_dataset.trans_meta_data.SGG_result_ind_to_classes,
            )

        traj_data_path = root + "traj_data.json"
        with open(traj_data_path, 'r') as f:
            traj_data = json.load(f)
        frames_depth = test_load_img(os.path.join(root, 'depth_images'), traj_data["images"], None).view(-1, 3, 300, 300)
        frames_rgb = test_load_img(os.path.join(root, 'instance_masks'), traj_data["images"], alfred_dataset.trans_meta_data.transforms).view(-1, 3, 300, 300)
        agent_meta_data = test_load_meta_data(root, traj_data["images"])
        for i in range(len(frames_depth)):
        # for i in range(3):
            depth_image = frames_depth[i]
            rgb_image = frames_rgb[i]
            agent_meta = agent_meta_data['agent_sgg_meta_data'][i]
            sgg_results = sgg_model.predict(rgb_image, 0)
            sgg_result = sgg_results[0]
            # import pdb; pdb.set_trace()
            target = {
                "bbox": sgg_result['bbox'],
                "labels": sgg_result['labels'],
            }
            # .view(-1, 300, 300, 3)
            cam_coords = grap_map.update_map(
                np.array(depth_image.view(300, 300, 3)),
                agent_meta,
                target)
            # import pdb; pdb.set_trace()
            grap_map.visualize_graph_map( np.array( sgg_result["write_img"] ), np.array(depth_image.view(300, 300, 3)))

            save_dir = "../visual_grap_map"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fn = save_dir + '/{}.png'.format(
                i)
            plt.savefig(fn)

        # grap_map.visualize_graph_map()
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