import os
import sys
import torch
from torch_geometric.data import Data
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.semantic_graph import utils
import importlib
from numpy import genfromtxt
import json
from icecream import ic
import numpy as np

__all__ = ['GraphData', 'SceneGraph', 'HeteGraphData']


class GraphData(Data):
    def __init__(self, obj_cls_name_to_features, GPU, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, device='cuda',**kwargs):
        super(GraphData, self).__init__(x=None, edge_index=None, edge_attr=None, y=None,
                                        pos=None, normal=None, face=None, **kwargs)
        # {'ButterKnife|-00.06|+01.11|+00.25': 0, 'Lettuce|-00.06|+01.19|-00.76': 1, 'Statue|+00.22|+01.11|-00.51': 2, 'CounterTop|-00.08|+01.15|00.00': 3, 'Knife|+00.22|+01.14|+00.25': 4, 'LightSwitch|+02.33|+01.31|-00.16': 5, 'DishSponge|-00.06|+01.11|-00.25': 6, 'Vase|-00.34|+01.11|+00.25': 7, 'Book|+00.16|+01.10|+00.62': 8}
        self.obj_id_to_ind = {}
        # {0: 'ButterKnife|-00.06|+01.11|+00.25', 1: 'Lettuce|-00.06|+01.19|-00.76', 2: 'Statue|+00.22|+01.11|-00.51', 3: 'CounterTop|-00.08|+01.15|00.00', 4: 'Knife|+00.22|+01.14|+00.25', 5: 'LightSwitch|+02.33|+01.31|-00.16', 6: 'DishSponge|-00.06|+01.11|-00.25', 7: 'Vase|-00.34|+01.11|+00.25', 8: 'Book|+00.16|+01.10|+00.62'}
        self.ind_to_obj_id = {}
        # {15: [0, 9, 10], 50: [1], 85: [2], 24: [3], 45: [4, 11], 52: [5], 30: [6], 102: [7], 10: [8]}
        self.obj_cls_to_ind = defaultdict(list)
        # list[0] = x[0] label
        # [15, 50, 85, 24, 45, 52, 30, 102, 10]
        self.list_node_obj_cls = []
        # edge
        self.edge_obj_to_obj = None
        # dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105])
        # self.obj_cls_to_features[1].shape 
        # torch.Size([300])
        self.obj_cls_to_features = obj_cls_name_to_features
        self.GPU = GPU
        self.device = device

    def __inc__(self, key, value):
        increasing_funcs = ["edge_obj_to_obj"]
        if key in increasing_funcs:
            return len(self["x"])
        else:
            return 0

    def add_node(self, obj_cls, feature, obj_id=None):
        '''
        obj_cls : 24 (int), object class index
        obj_id : StoveKnob|-03.60|+01.11|+01.87
        feature : tensor size 23, tensor([1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.], requires_grad=True)

        return the index of the node
        '''
        # shape : 300 + 23
        feature = self._cat_feature_and_wore_embed(obj_cls, feature).unsqueeze(0)
        # init self.x
        if self.x is None:
            ind = 0
            self.x = feature
        else:
            ind = len(self.x)
            self.x = torch.cat([self.x, feature], dim=0)
        # Oracle
        if obj_id is None:
            # Not Oracle. Create new obj_id
            obj_id = str(obj_cls) + "_" + str(len(self.obj_cls_to_ind[obj_cls]))
        self.obj_id_to_ind[obj_id] = ind
        self.ind_to_obj_id[ind] = obj_id
        self.obj_cls_to_ind[obj_cls].append(ind)
        self.list_node_obj_cls.append(obj_cls)
        return ind

    def update_node(self, obj_cls, feature, obj_id):
        '''
        obj_cls : 24, object class index
        obj_id : StoveKnob|-03.60|+01.11|+01.87

        return the index of the node
        '''
        # shape : 300 + 23
        isFeatureChange = False
        feature = self._cat_feature_and_wore_embed(obj_cls, feature)
        ind = self.obj_id_to_ind[obj_id]
        if not torch.equal(self.x[ind], feature):
            isFeatureChange = True
        self.x[ind] = feature
        return ind, isFeatureChange

    def update_relation(self, node_src, node_dst, relation):
        '''
        node_src : 320
        node_dst : 321
        relation : 1
        '''
        tensor_node = torch.tensor([[node_src], [node_dst]], dtype=torch.long).contiguous()
        if self.GPU:
            tensor_node = tensor_node.to(self.device)
        if self.edge_obj_to_obj is None:
            self.edge_obj_to_obj = tensor_node
        else:
            '''
            tensor([[ 5,  0],
                    [ 9,  0],
                    [ 2,  1],
                    [ 8,  1],
                    [ 8,  2],
                    [ 8,  6],
                    [ 8,  7],
                    [ 5,  9],
                    [ 6, 10],
                    [ 8, 10]])
            '''
            self.edge_obj_to_obj = torch.cat([self.edge_obj_to_obj, tensor_node], dim=1)

    def search_node(self, obj_id=None):
        # Oracle
        if obj_id is not None:
            if obj_id in self.obj_id_to_ind:
                return True
            else:
                return False
        # compare alg
        else:
            raise NotImplementedError

    def get_wore_embed(self, obj_cls):
        word_embed = self.obj_cls_to_features[obj_cls].clone().detach()
        if self.GPU:
            word_embed = word_embed.to(self.device)
        return word_embed

    def _cat_feature_and_wore_embed(self, obj_cls, feature):
        feature = feature.clone().detach()
        if self.GPU:
            feature = feature.to(self.device)
        word_embed = self.get_wore_embed(obj_cls)
        feature = torch.cat([feature, word_embed])
        return feature


class HeteGraphData(GraphData):
    def __init__(self, obj_cls_name_to_features, GPU, dim_rgb_feature, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, normal=None, face=None, device="cuda", **kwargs):
        super(HeteGraphData, self).__init__(obj_cls_name_to_features, GPU,
                                            x=None, edge_index=None, edge_attr=None, y=None,
                                            pos=None, normal=None, face=None, device=device, **kwargs)
        self.attributes = None
        self.dim_rgb_feature = dim_rgb_feature

    def __inc__(self, key, value):
        increasing_funcs = ["edge_obj_to_obj"]
        if key in increasing_funcs:
            return len(self["x"])
        else:
            return 0

    def _append_unique_obj_index_to_attribute(self, feature, obj_cls):
        unique_obj_index = len(self.obj_cls_to_ind[obj_cls])
        unique_obj_index_tensor = torch.tensor([unique_obj_index], dtype=torch.float)
        feature = torch.cat([feature, unique_obj_index_tensor]).unsqueeze(0)
        if self.GPU:
            feature = feature.to(self.device)
        return feature, unique_obj_index

    def add_node(self, obj_cls, attr_feature, obj_id=None, feature_img=None):
        node_feature = self.get_wore_embed(obj_cls).unsqueeze(0)
        attr_feature, unique_obj_index = self._append_unique_obj_index_to_attribute(attr_feature, obj_cls)

        if feature_img is not None:
            if self.GPU:
                feature_img = feature_img.to(self.device)
            node_feature = torch.cat([node_feature, feature_img], dim=1)
        # init self.x
        if self.x is None:
            ind = 0
            self.x = node_feature
            self.attributes = attr_feature
        else:
            ind = len(self.x)
            self.x = torch.cat([self.x, node_feature], dim=0)
            self.attributes = torch.cat([self.attributes, attr_feature], dim=0)
        # Oracle
        if obj_id is None:
            # Not Oracle. Create new obj_id
            obj_id = str(obj_cls) + "_" + str(unique_obj_index)
        self.obj_id_to_ind[obj_id] = ind
        self.ind_to_obj_id[ind] = obj_id
        self.obj_cls_to_ind[obj_cls].append(ind)
        self.list_node_obj_cls.append(obj_cls)
        return ind

    def update_node(self, obj_cls, attr_feature, obj_id, feature_img=None):
        '''
        isFeatureChange: if True, add to history graph
        '''
        isFeatureChange = False
        ind = self.obj_id_to_ind[obj_id]
        if self.GPU:
            attr_feature = attr_feature.to(self.device)
            if feature_img is not None:
                feature_img = feature_img.to(self.device)
        if not torch.equal(self.attributes[ind][:-1], attr_feature):
            isFeatureChange = True
        # set feature
        self.attributes[ind][:-1] = attr_feature
        if feature_img is not None:
            # from -512
            self.x[ind][-self.dim_rgb_feature:] = feature_img
        return ind, isFeatureChange

    def update_adj_relation(self, node_src_ind, obj_cls, adj, relation_type=1):
        '''
        obj_cls: int, 1~105
        candidate_nodes_relation: [0, 1, 1, ..., 1, 0]
        '''
        candidate_nodes_relation = adj[obj_cls]
        candidate_obj_cls_has_relation_list = np.where(candidate_nodes_relation == 1)[0]
        for candidate_obj_cls in candidate_obj_cls_has_relation_list:
            # check if node_dst in graph
            for obj_cls_to_ind_node_dst_ind in self.obj_cls_to_ind[candidate_obj_cls]:
                self.update_relation(node_src_ind, obj_cls_to_ind_node_dst_ind, relation_type)
                self.update_relation(obj_cls_to_ind_node_dst_ind, node_src_ind, relation_type)

    def update_relation(self, node_src, node_dst, relation):
        '''
        node_src : 320, node index in this graph
        node_dst : 321, node index in this graph
        relation : 1
        '''
        tensor_node = torch.tensor([[node_src], [node_dst]], dtype=torch.long).contiguous()
        if self.GPU:
            tensor_node = tensor_node.to(self.device)
        if self.edge_obj_to_obj is None:
            self.edge_obj_to_obj = tensor_node
        else:
            '''
            tensor([[ 5,  0],
                    [ 9,  0],
                    [ 2,  1],
                    [ 8,  1],
                    [ 8,  2],
                    [ 8,  6],
                    [ 8,  7],
                    [ 5,  9],
                    [ 6, 10],
                    [ 8, 10]])
            '''
            self.edge_obj_to_obj = torch.cat([self.edge_obj_to_obj, tensor_node], dim=1)

    def search_node(self, obj_id=None):
        # Oracle
        if obj_id is not None:
            if obj_id in self.obj_id_to_ind:
                return True
            else:
                return False
        # compare alg
        else:
            raise NotImplementedError

    def _cat_feature_and_wore_embed(self, obj_cls, feature):
        raise NotImplementedError


'''
For store the node & relation of global_graph, current_state_graph, history_changed_nodes_graph, priori_graph
'''
class SceneGraph(object):
    """docstring for SceneGraph"""

    def __init__(self, cfg, object_classes_index_to_name, dim_rgb_feature, device="cuda"):
        super(SceneGraph, self).__init__()
        #
        self.device = device
        self.cfg = cfg
        self.isORACLE = cfg.SCENE_GRAPH.ORACLE
        self.GPU = cfg.SCENE_GRAPH.GPU
        self.GRAPH_RESULT_PATH = cfg.SCENE_GRAPH.GRAPH_RESULT_PATH
        self.RELATION_MODE = cfg.SCENE_GRAPH.RELATION_MODE if "RELATION_MODE" in cfg.SCENE_GRAPH else 0.7
        self.ANGLE_OF_VIEWS = cfg.SCENE_GRAPH.ANGLE_OF_VIEWS if "ANGLE_OF_VIEWS" in cfg.SCENE_GRAPH else False
        '''
        # SGG vision
        '''
        self.SAME_VISION_FEATURE_THRESHOLD = cfg.SGG.SAME_VISION_FEATURE_THRESHOLD\
            if "SGG" in cfg and "SAME_VISION_FEATURE_THRESHOLD" in cfg.SGG else cfg.SCENE_GRAPH.SAME_VISION_FEATURE_THRESHOLD
        self.SEARCH_EXISTING_NODE_METHOD = cfg.SGG.SEARCH_EXISTING_NODE_METHOD\
            if "SGG" in cfg and "SEARCH_EXISTING_NODE_METHOD" in cfg.SGG else "BASIC"
        # object class name
        self.object_classes_index_to_name = object_classes_index_to_name

        '''
        Load node data feature & relation
        '''
        # graph feature
        path_object_embedding = self.cfg.SCENE_GRAPH.OBJ_NAME_EMBEDDING
        path_object_embedding = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), path_object_embedding)
        path_ADJ = self.cfg.SCENE_GRAPH.PRIORI_OBJ_ADJ
        path_ADJ_file = os.path.join(os.environ['GRAPH_ANALYSIS'], path_ADJ)
        self.obj_cls_name_to_features = self._get_obj_cls_name_to_features(path_object_embedding, _background=True)
        adj = genfromtxt(path_ADJ_file, delimiter=',').astype(int)
        self.adj = np.zeros((adj.shape[0]+1, adj.shape[0]+1))
        self.adj[1:, 1:] = adj

        '''
        # create graph
        '''
        self.dim_rgb_feature = dim_rgb_feature
        self.graphdata_type = getattr(
            importlib.import_module(
                'agents.semantic_graph.semantic_graph'),
            self.cfg.SCENE_GRAPH.GraphData
            )
        ic(self.cfg.SCENE_GRAPH.GraphData)
        self.init_graph_data()
        # priori_graph
        if "PRIORI" in cfg.SCENE_GRAPH and cfg.SCENE_GRAPH.PRIORI:
            self._set_priori_graph()

        assert len(self.adj) == len(self.object_classes_index_to_name)
        assert len(self.obj_cls_name_to_features) == len(self.object_classes_index_to_name)-1, \
            "object_classes_index_to_name object list must be same with obj_cls_name_to_features because add_oracle_local_graph_to_global_graph need object_classes feature. \
                Please read alfworld/agents/semantic_graph/word_embed/README.md"

    def init_graph_data(self):
        self.global_graph = self.graphdata_type(self.obj_cls_name_to_features, self.GPU, self.dim_rgb_feature, device=self.device)
        self.current_state_graph = self.graphdata_type(self.obj_cls_name_to_features, self.GPU, self.dim_rgb_feature, device=self.device)
        self.history_changed_nodes_graph = self.graphdata_type(self.obj_cls_name_to_features, self.GPU, self.dim_rgb_feature, device=self.device)

    # adjacency_matrix_and_all_feature
    def _set_priori_graph(self):
        path_object_embedding = self.cfg.SCENE_GRAPH.PRIORI_OBJ_NAME_EMBEDDING
        path_object_embedding = os.path.join(os.environ['GRAPH_ANALYSIS'], path_object_embedding)
        path_object_rgb_feature = self.cfg.SCENE_GRAPH.PRIORI_OBJ_RBG_FEATURE_EMBEDDING
        path_object_rgb_feature = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), path_object_rgb_feature)
        path_object_attribute = self.cfg.SCENE_GRAPH.PRIORI_OBJ_ATTRIBUTE_FEATURE
        path_object_attribute = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), path_object_attribute)

        with open(path_object_rgb_feature, 'r') as f:
            rgb_features = json.load(f)
        with open(path_object_attribute, 'r') as f:
            attributes = json.load(f)
        priori_obj_cls_name_to_features = self._get_obj_cls_name_to_features(path_object_embedding, _background=True)
        self.priori_graph = self.graphdata_type(priori_obj_cls_name_to_features, self.GPU, self.dim_rgb_feature, device=self.device)

        ic("Prior visual feature: ", len(rgb_features["0"]))
        ic("NODE_INPUT_RGB_FEATURE_SIZE: ", self.dim_rgb_feature)
        assert len(rgb_features["0"]) == self.dim_rgb_feature, "shape must be same, else Graph Net downsample would have error"
        # node word feature
        for k, word_feature in priori_obj_cls_name_to_features.items():
            '''
            rgb_features: "0"~"105"
            attributes: "0"~"105"
            k: 1~105 (avoid background=True)
            word_feature: shape = [300]
            self.priori_graph.x index from 0~104
            '''
            # rgb feature
            rgb_feature = [rgb_features[str(k)]]
            rgb_feature = torch.tensor(rgb_feature).float()
            attribute = torch.tensor(attributes[str(k)]).float()
            self.priori_graph.add_node(
                obj_cls=k,
                attr_feature=attribute,
                feature_img=rgb_feature
                )
        # node relation
        # import pdb; pdb.set_trace()
        assert len(priori_obj_cls_name_to_features) == len(self.adj)-1
        adj = self.adj
        for src in range(1, len(adj)):
            for dst in range(1, len(adj)):
                if adj[src, dst] > 0:
                    self.priori_graph.update_relation(src-1, dst-1, 0)
        print("need to check word & rgb feature & relation is ok")
        self.priori_features = dict(
            priori_obj_cls_name_to_features=priori_obj_cls_name_to_features,
            rgb_features=rgb_features,
            attributes=attributes,
            )

    def init_current_state_data(self):
        self.current_state_graph = self.graphdata_type(self.obj_cls_name_to_features, self.GPU, self.dim_rgb_feature, device=self.device)

    def _get_obj_cls_name_to_features(self, path, _background=False):
        def get_feature(csv_nodes_data):
            feature = [csv_nodes_data['feature'].to_list()]
            for i in range(1, 300):
                feature.extend([csv_nodes_data['feature.{}'.format(i)].to_list()])
            feature = torch.tensor(feature).float().transpose(0, 1)
            obj_cls_name_to_features = {}
            for i, obj_cls_name in enumerate(csv_nodes_data['Id'].to_list()):
                if _background:
                    # for avoid class 0 __background__
                    obj_cls_name += 1
                obj_cls_name_to_features[obj_cls_name] = feature[i]
            return obj_cls_name_to_features
        csv_obj_features = pd.read_csv(path)
        obj_cls_name_to_features = get_feature(csv_obj_features)
        return obj_cls_name_to_features

    def get_graph_data(self):
        # print("global_graph")
        return self.global_graph

    def get_history_changed_nodes_graph_data(self):
        # print("history_changed_nodes_graph")
        return self.history_changed_nodes_graph

    def get_current_state_graph_data(self):
        # print("current_state_graph")
        return self.current_state_graph

    def get_priori_graph(self):
        # print("priori_graph")
        return self.priori_graph

    def analyze_graph(self, dict_ANALYZE_GRAPH, graph_type="GLOBAL_GRAPH"):
        '''
        dict_ANALYZE_GRAPH = {
            "score": score.clone().detach().to('cpu'),
            "sort_nodes_index": sort_nodes_index.clone().detach().to('cpu'),
        }
        '''
        if graph_type == "GLOBAL_GRAPH":
            graph = self.global_graph
        elif graph_type == "CURRENT_STATE_GRAPH":
            graph = self.current_state_graph
        elif graph_type == "HISTORY_CHANGED_NODES_GRAPH":
            graph = self.history_changed_nodes_graph
        elif graph_type == "PRIORI_GRAPH":
            graph = self.priori_graph
        else:
            raise NotImplementedError

        dict_objectIds_to_score = {}
        try:
            for i, nodes_index in enumerate(dict_ANALYZE_GRAPH["sort_nodes_index"]):
                obj_cls = graph.list_node_obj_cls[nodes_index]
                obj_name = self.object_classes_index_to_name[obj_cls]
                score = np.round(dict_ANALYZE_GRAPH["score"][i], decimals=5)
                dict_objectIds_to_score[obj_name] = score
        except Exception as e:
            pass
        return dict_objectIds_to_score

    def save_graph_data(self):
        if not os.path.exists(self.GRAPH_RESULT_PATH):
            os.mkdir(self.GRAPH_RESULT_PATH)
        utils.save_graph_data(self.global_graph, self.GRAPH_RESULT_PATH)

    # Imagehash https://content-blockchain.org/research/testing-different-image-hash-functions/
    # https://pypi.org/project/ImageHash/
    def compare_global_graph_existing_node(self, obj_cls, feature):
        '''
        obj_cls: object class
        feature: visual feature (sgg feature)

        obj_id: obj_id in global graph
        node_ind: obj_cls_to_ind in global graph
        isFindNode: similarity >= self.SAME_VISION_FEATURE_THRESHOLD
        '''
        obj_id = None
        nodes = self.global_graph.x
        obj_cls_to_ind = self.global_graph.obj_cls_to_ind
        for suspect_node_ind in obj_cls_to_ind[obj_cls]:
            suspect_node_feature = nodes[suspect_node_ind]
            suspect_node_feature = suspect_node_feature[-self.dim_rgb_feature:].clone().to('cpu')
            similarity = self.compare_features(feature, suspect_node_feature)[0,0]
            # print("similarity: ", similarity)
            if similarity >= self.SAME_VISION_FEATURE_THRESHOLD:
                # node ind to obj_id
                obj_id = self.global_graph.ind_to_obj_id[suspect_node_ind]
                return obj_id, suspect_node_ind, True
        return obj_id, None, False

    def compare_features(self, feature1, feature2):
        return cosine_similarity(feature1.reshape(1, -1), feature2.reshape(1, -1))

    def add_adj_relation(self, graph, current_frame_obj_cls_to_node_index, add_relation_anyway=False):
        '''
        relation_have_been_add: isFindNode
        '''
        for node_ind, obj_cls, relation_have_been_add in current_frame_obj_cls_to_node_index:
            if not relation_have_been_add or add_relation_anyway:
                graph.update_adj_relation(node_ind, obj_cls, self.adj)

    def add_ai2_thor_relation(self, graph, current_frame_obj_cls_to_node_index, obj_relations):
        '''
        current_frame_obj_cls_to_node_index: [(ind, obj_cls, isFindNode), ...]
        '''
        for obj_relation_triplet in obj_relations:
            src_obj_ind, dst_obj_ind, relation = obj_relation_triplet
            src_node_ind = current_frame_obj_cls_to_node_index[src_obj_ind][0]
            dst_node_ind = current_frame_obj_cls_to_node_index[dst_obj_ind][0]
            if src_node_ind is not None and dst_node_ind is not None:
                graph.update_relation(src_node_ind, dst_node_ind, relation)

    def add_oracle_local_graph_to_global_graph(self, feature_img, target, reset_current_graph=True):
        '''
        feature_img: torch.Size([512, 7, 7])
        '''
        if reset_current_graph:
            self.init_current_state_data()
        # global average pooling, https://discuss.pytorch.org/t/how-can-i-perform-global-average-pooling-before-the-last-fully-connected-layer/74352
        if len(feature_img.shape) == 3:
            feature_img = feature_img.mean([1, 2]).reshape(1, -1)
        else:
            feature_img = feature_img.reshape(1, -1)
        tar_ids = target["objectIds"]
        obj_clses = target["labels"].numpy().astype(int)
        obj_attributes = target["attributes"]
        obj_angle_of_views = target["angle_of_views"]
        # other way to process (index 4 is out of bounds for dimension 0 with size 4)
        obj_relations = target["relation_labels"].numpy().astype(int)
        global_graph_current_frame_obj_cls_to_node_index = []
        current_state_graph_current_frame_obj_cls_to_node_index = []

        '''
        tar_ids : ['Fridge|-01.50|+00.00|-00.70', 'Bread|+00.40|+00.96|+00.11|BreadSliced_6', 'Mug|-01.28|+01.01|-01.64', 'Microwave|-01.23|+00.90|-01.68', 'Cabinet|-00.84|+00.47|-01.67', 'CounterTop|+00.23|+00.95|-02.00', 'Drawer|-00.82|+00.75|-01.69']
        obj_cls : tensor(24.), object class index
        '''
        for i, tar_id in enumerate(tar_ids):
            obj_cls, obj_attribute, obj_angle_of_view = \
                obj_clses[i], obj_attributes[i], obj_angle_of_views[i]
            if self.ANGLE_OF_VIEWS:
                obj_attribute = torch.cat([obj_attribute, obj_angle_of_view])
            isFindNode = self.global_graph.search_node(obj_id=tar_id)
            # check node
            if isFindNode:
                ind, isFeatureChange = self.global_graph.update_node(obj_cls, obj_attribute, tar_id, feature_img=feature_img)
                # EMBED_HISTORY_CHANGED_NODES
                if isFeatureChange:
                    self.history_changed_nodes_graph.add_node(obj_cls, obj_attribute, tar_id, feature_img=feature_img)
            else:
                ind = self.global_graph.add_node(obj_cls, obj_attribute, tar_id, feature_img=feature_img)
            # EMBED_CURRENT_STATE
            current_state_node_ind = self.current_state_graph.add_node(obj_cls, obj_attribute, tar_id, feature_img=feature_img)
            # [317, 318, 148, 149, 150, 151, 152, 319, 320, 321, 322, 154, 323, 155, 157, 158, 159, 160, 161, 324]
            global_graph_current_frame_obj_cls_to_node_index.append((ind, obj_cls, isFindNode))
            current_state_graph_current_frame_obj_cls_to_node_index.append((current_state_node_ind, obj_cls, isFindNode))
        '''
        # relation
        '''
        # no matter isFindNode is True or False,
        # self.current_state_graph always need to add adj relation
        # if not isFindNode:
        self.add_ai2_thor_relation(self.current_state_graph,
                                   current_state_graph_current_frame_obj_cls_to_node_index,
                                   obj_relations
                                   )
        self.add_ai2_thor_relation(self.global_graph,
                                   global_graph_current_frame_obj_cls_to_node_index,
                                   obj_relations
                                   )

        # self.add_adj_relation(self.current_state_graph,
        #                       current_state_graph_current_frame_obj_cls_to_node_index,
        #                       add_relation_anyway=True
        #                       )
        # self.add_ai2_thor_relation(self.global_graph,
        #                            global_graph_current_frame_obj_cls_to_node_index,
        #                            obj_relations
        #                            )

    def add_local_graph_to_global_graph(self, img, sgg_results, reset_current_graph=True):
        if "JACCARD" == self.SEARCH_EXISTING_NODE_METHOD:
            current_state_graph_current_frame_obj_cls_to_node_index, global_graph_current_frame_obj_cls_to_node_index = \
                self._jaccard(img, sgg_results, reset_current_graph)
        else:
            current_state_graph_current_frame_obj_cls_to_node_index, global_graph_current_frame_obj_cls_to_node_index = \
                self._basic(img, sgg_results, reset_current_graph)

        # tensor([[9.9977e-01, 2.3437e-04],
        #         [2.1592e-01, 7.8408e-01],
        #         [2.6653e-01, 7.3347e-01],
        #         [6.4093e-01, 3.5907e-01], ...
        obj_relations_scores = sgg_results["obj_relations_scores"]
        # array([[ 0,  1],
        #        [ 0,  2],
        #        [ 0,  3],
        #        [ 0,  4], ...
        obj_relations_idx_pairs = sgg_results["obj_relations_idx_pairs"].numpy().astype(int)
        obj_relations = []
        for i, max_relation in enumerate(torch.argmax(obj_relations_scores, dim=1).numpy()):
            # relation sgg: "0" no relation. "1" ai2-thor relation
            if max_relation == 1:
                # src_obj_ind, dst_obj_ind = obj_relations_idx_pairs[i]
                # obj_relation_triplet = src_obj_ind, dst_obj_ind, relation
                obj_relation_triplet = obj_relations_idx_pairs[i].tolist() + [max_relation]
                obj_relations.append(obj_relation_triplet)
                # debug
                # src_obj_ind, dst_obj_ind = obj_relations_idx_pairs[i]
                # src_node_ind = global_graph_current_frame_obj_cls_to_node_index[src_obj_ind]
                # dst_node_ind = global_graph_current_frame_obj_cls_to_node_index[dst_obj_ind]
                # print("src_node_ind {}, dst_node_ind {}, relation {}".format(src_node_ind, dst_node_ind, max_relation))

        self.add_ai2_thor_relation(self.current_state_graph,
                                   current_state_graph_current_frame_obj_cls_to_node_index,
                                   obj_relations
                                   )
        self.add_ai2_thor_relation(self.global_graph,
                                   global_graph_current_frame_obj_cls_to_node_index,
                                   obj_relations
                                   )

    def _basic(self, img, sgg_results, reset_current_graph=True):
        if reset_current_graph:
            self.init_current_state_data()
        global_graph_current_frame_obj_cls_to_node_index, current_state_graph_current_frame_obj_cls_to_node_index = [], []
        obj_clses = sgg_results["labels"].numpy().astype(int)
        # torch.Size([16, 2048, 1, 1])
        obj_features = sgg_results["features"]
        # torch.Size([16, 23])
        obj_attributes = sgg_results["attribute_logits"]
        # import pdb; pdb.set_trace()
        for i, obj_cls in enumerate(obj_clses):
            obj_feature = obj_features[i].reshape(1, -1)
            obj_attribute = obj_attributes[i].reshape(-1)
            obj_id, exist_node_ind, isFindNode = self.compare_global_graph_existing_node(obj_cls, obj_feature)
            if obj_id is not None:
                ind, _ = self.global_graph.update_node(obj_cls, obj_attribute, obj_id, feature_img=obj_feature)
            else:
                ind = self.global_graph.add_node(obj_cls, obj_attribute, feature_img=obj_feature)
            current_state_node_ind = self.current_state_graph.add_node(obj_cls, obj_attribute, feature_img=obj_feature)

            global_graph_current_frame_obj_cls_to_node_index.append((ind, obj_cls, isFindNode))
            current_state_graph_current_frame_obj_cls_to_node_index.append((current_state_node_ind, obj_cls, isFindNode))

        return current_state_graph_current_frame_obj_cls_to_node_index, global_graph_current_frame_obj_cls_to_node_index

    def _jaccard(self, img, sgg_results, reset_current_graph=True):
        def jaccard_similarity(pre_graph, curr_graph):
            # [15, 50, 85, 24, 45, 52, 3, 5]
            p_graph_object_list = pre_graph.list_node_obj_cls
            # [15, 50, 85, 24, 45, 52, 30, 102, 10, 10]
            c_graph_object_list = curr_graph.list_node_obj_cls
            # [30, 102, 10, 10]
            return list((Counter(c_graph_object_list) - Counter(p_graph_object_list)).elements())
        self.pre_current_state_graph = self.current_state_graph
        if reset_current_graph:
            self.init_current_state_data()
        global_graph_current_frame_obj_cls_to_node_index, current_state_graph_current_frame_obj_cls_to_node_index = [], []
        obj_clses = sgg_results["labels"].numpy().astype(int)
        # torch.Size([16, 2048, 1, 1])
        obj_features = sgg_results["features"]
        # torch.Size([16, 23])
        obj_attributes = sgg_results["attribute_logits"]
        # import pdb; pdb.set_trace()
        for i, obj_cls in enumerate(obj_clses):
            obj_feature = obj_features[i].reshape(1, -1)
            obj_attribute = obj_attributes[i].reshape(-1)
            current_state_node_ind = self.current_state_graph.add_node(obj_cls, obj_attribute, feature_img=obj_feature)
            current_state_graph_current_frame_obj_cls_to_node_index.append((current_state_node_ind, obj_cls, False))
        extra_object = jaccard_similarity(self.pre_current_state_graph, self.current_state_graph)
        for i, obj_cls in enumerate(obj_clses):
            obj_feature = obj_features[i].reshape(1, -1)
            obj_attribute = obj_attributes[i].reshape(-1)
            obj_id, exist_node_ind, isFindNode = self.compare_global_graph_existing_node(obj_cls, obj_feature)
            # [30, 102, 10, 10]
            if obj_cls in extra_object:
                if obj_id is not None:
                    ind, _ = self.global_graph.update_node(obj_cls, obj_attribute, obj_id, feature_img=obj_feature)
                else:
                    ind = self.global_graph.add_node(obj_cls, obj_attribute, feature_img=obj_feature)
            else:
                if exist_node_ind is not None:
                    ind = exist_node_ind
                else:
                    ind = None
            global_graph_current_frame_obj_cls_to_node_index.append((ind, obj_cls, isFindNode))
        return current_state_graph_current_frame_obj_cls_to_node_index, global_graph_current_frame_obj_cls_to_node_index


if __name__ == '__main__':
    sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
    sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
    import modules.generic as generic
    from sgg import alfred_data_format

    cfg = generic.load_config()
    cfg = cfg['semantic_cfg']
    scenegraph = SceneGraph(cfg)
    # trans_meta_data = alfred_data_format.TransMetaData(cfg)
    alfred_dataset = alfred_data_format.AlfredDataset(cfg)
    for i in range(10):
        img, target, idx = alfred_dataset[i]
        dict_target = {
            "labels": target.extra_fields["labels"],
            "obj_relations": target.extra_fields["pred_labels"],
            "relation_labels": target.extra_fields["relation_labels"],
            "attributes": target.extra_fields["attributes"],
            "objectIds": target.extra_fields["objectIds"],
        }
        scenegraph.add_oracle_local_graph_to_global_graph(img, dict_target)
        scenegraph.save_graph_data()
