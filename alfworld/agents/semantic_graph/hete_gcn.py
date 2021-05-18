import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import graph_embed


# https://github.com/rusty1s/pytorch_geometric/issues/1083
class Net(torch.nn.Module):
    """docstring for Net"""
    def __init__(self, cfg, config=None, PRINT_DEBUG=False, device='cuda'):
        super(Net, self).__init__()
        input_size = cfg.SCENE_GRAPH.NODE_MIDDEL_FEATURE_SIZE
        middle_size = cfg.SCENE_GRAPH.NODE_MIDDEL_FEATURE_SIZE
        ATTRIBUTE_FEATURE_SIZE = cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE
        output_size = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE
        # True => one of the variables needed for gradient computation has been modified by an inplace operation
        normalize = cfg.SCENE_GRAPH.NORMALIZATION
        self.device = device
        self.cfg = cfg
        self.PRINT_DEBUG = PRINT_DEBUG
        self.node_word_embed_downsample = nn.Linear(cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE, middle_size//2)
        self.node_rgb_feature_downsample = nn.Linear(cfg.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE, middle_size//2)
        self.conv1 = GCNConv(input_size, middle_size, cached=not normalize,
                             normalize=normalize,
                             )
        self.conv2 = GCNConv(middle_size + ATTRIBUTE_FEATURE_SIZE, output_size, cached=not normalize,
                             normalize=normalize,
                             )
        self.dropout = nn.Dropout()
        # self.attri_linear = nn.Linear(ATTRIBUTE_FEATURE_SIZE, ATTRIBUTE_FEATURE_SIZE
        #                               )
        graph_embed_model = getattr(graph_embed, cfg.SCENE_GRAPH.EMBED_TYPE)
        NODE_FEATURE_SIZE = cfg.SCENE_GRAPH.NODE_OUT_FEATURE_SIZE + cfg.SCENE_GRAPH.ATTRIBUTE_FEATURE_SIZE
        NUM_CHOSE_NODE = cfg.SCENE_GRAPH.NUM_CHOSE_NODE
        self.EMBED_FEATURE_SIZE = cfg.SCENE_GRAPH.EMBED_FEATURE_SIZE
        self.final_mapping = graph_embed_model(
            INPUT_FEATURE_SIZE=NODE_FEATURE_SIZE,
            EMBED_FEATURE_SIZE=self.EMBED_FEATURE_SIZE,
            NUM_CHOSE_NODE=NUM_CHOSE_NODE,
        )
        if cfg.SCENE_GRAPH.CHOSE_IMPORTENT_NODE:
            # nn.linear bert_hidden_size -> NODE_FEATURE_SIZE
            bert_hidden_size = config['general']['model']['block_hidden_dim']
            self.CHOSE_IMPORTENT_NODE_OUTPUT_SHAPE = NODE_FEATURE_SIZE * NUM_CHOSE_NODE
            self.chose_node_module = graph_embed.DotAttnChoseImportentNode(
                bert_hidden_size,
                NODE_FEATURE_SIZE,
                NUM_CHOSE_NODE,
                cfg.SCENE_GRAPH.GPU,
                PRINT_DEBUG=PRINT_DEBUG
            )
            self.chose_node_feature_mapping = nn.Linear(
                self.CHOSE_IMPORTENT_NODE_OUTPUT_SHAPE,
                self.EMBED_FEATURE_SIZE
            )
        self.to(device)

    def forward(self, data, CHOSE_IMPORTENT_NODE=False, hidden_state=None):
        '''
        data.x
        tensor([[-0.0474,  0.0324,  0.1443,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0440, -0.0058,  0.0014,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0057,  0.0471,  0.0377,  ...,  1.0000,  0.0000,  0.0000],
                [ 0.0724, -0.0065, -0.0210,  ...,  0.0000,  0.0000,  0.0000],
                [-0.0474,  0.0324,  0.1443,  ...,  1.0000,  0.0000,  0.0000]],
               grad_fn=<CatBackward>)
        data.edge_obj_to_obj
        tensor([[3, 0],
                [3, 1],
                [3, 2],
                [3, 4]])
        data.obj_cls_to_ind
        {64: [0, 4], 70: [1], 47: [2], 81: [3]}
        data.obj_id_to_ind
        {'Pillow|-02.89|+00.62|+00.82': 0, 'RemoteControl|-03.03|+00.56|+02.01': 1, 'Laptop|-02.81|+00.56|+01.81': 2, 'Sofa|-02.96|+00.08|+01.39': 3, 'Pillow|-02.89|+00.62|+01.19': 4}
        '''
        # import pdb; pdb.set_trace()
        x, attributes, edge_obj_to_obj, edge_weight = \
            data.x, \
            data.attributes, \
            data.edge_obj_to_obj, \
            data.edge_attr
        dict_ANALYZE_GRAPH = None
        # no edge, without gcn
        if edge_obj_to_obj is None:
            # if self.PRINT_DEBUG:
            #     print("WARNING edge_obj_to_obj is None")
            #     if attributes is None:
            #         print("WARNING attributes is None")
            # have node
            if attributes is not None:
                # torch.Size([10, 300])
                x = x.clone().detach()
                # torch.Size([10, 24])
                attributes = attributes.clone().detach()
                word_feature = self.node_word_embed_downsample(x[:, :self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE])
                rgb_feature = self.node_rgb_feature_downsample(x[:, self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE:])
                # torch.Size([10, 16])
                x = torch.cat([word_feature, rgb_feature, attributes], dim=1)
                x = F.relu(self.dropout(x))
                nodes_tensor = x
                x, dict_ANALYZE_GRAPH = self.final_mapping(x)
                x = F.relu(self.dropout(x))
                if CHOSE_IMPORTENT_NODE:
                    chose_nodes, dict_ANALYZE_GRAPH = self.chose_node_module(nodes_tensor, hidden_state)
                    x = torch.cat([x, chose_nodes], dim=1)
            # don't have node
            else:
                x = torch.zeros((1, self.EMBED_FEATURE_SIZE))
                if CHOSE_IMPORTENT_NODE:
                    # torch.Size([1, 400])
                    chose_nodes = torch.zeros((1, self.CHOSE_IMPORTENT_NODE_OUTPUT_SHAPE))
                    x = torch.cat([x, chose_nodes], dim=1)
                if self.cfg.SCENE_GRAPH.GPU:
                    x = x.to(self.device)
        else:
            x = x.clone().detach()
            attributes = attributes.clone().detach()
            edge_obj_to_obj = edge_obj_to_obj.clone().detach()
            word_feature = self.node_word_embed_downsample(x[:, :self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE])
            rgb_feature = self.node_rgb_feature_downsample(x[:, self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE:])
            # torch.Size([2, 16])
            x = torch.cat([word_feature, rgb_feature], dim=1)
            x = F.relu(self.dropout(x))
            x = F.relu(self.conv1(x, edge_obj_to_obj, edge_weight))
            x = self.dropout(x)
            x = torch.cat([x, attributes], dim=1)
            # torch.Size([2, 40])
            x = self.conv2(x, edge_obj_to_obj, edge_weight)
            x = F.relu(self.dropout(x))
            # torch.Size([2, 16]) + torch.Size([2, 24]) => torch.Size([2, 40])
            x = torch.cat([x, attributes], dim=1)
            nodes_tensor = x

            # torch.Size([1, 128])
            x, dict_ANALYZE_GRAPH = self.final_mapping(x)
            x = F.relu(self.dropout(x))
            if CHOSE_IMPORTENT_NODE:
                # torch.Size([1, 400])
                chose_nodes, dict_ANALYZE_GRAPH = self.chose_node_module(nodes_tensor, hidden_state)
                # torch.Size([1, 528]) = self.cfg.SCENE_GRAPH.RESULT_FEATURE
                x = torch.cat([x, chose_nodes], dim=1)
        return x, dict_ANALYZE_GRAPH

    def chose_importent_node(self, data, hidden_state):
        x, attributes, edge_obj_to_obj, edge_weight = \
            data.x, \
            data.attributes, \
            data.edge_obj_to_obj, \
            data.edge_attr
        dict_ANALYZE_GRAPH = None
        # no edge, without gcn
        if edge_obj_to_obj is None:
            # if self.PRINT_DEBUG:
            #     print("WARNING edge_obj_to_obj is None")
            #     if attributes is None:
            #         print("WARNING attributes is None")
            # have node
            if attributes is not None:
                # torch.Size([10, 300])
                x = x.clone().detach()
                # torch.Size([10, 24])
                attributes = attributes.clone().detach()
                word_feature = self.node_word_embed_downsample(x[:, :self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE])
                rgb_feature = self.node_rgb_feature_downsample(x[:, self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE:])
                # torch.Size([10, 16]) + [10, 24]
                x = torch.cat([word_feature, rgb_feature, attributes], dim=1)
                x = F.relu(self.dropout(x))
                nodes_tensor = x
                chose_nodes, dict_ANALYZE_GRAPH = self.chose_node_module(nodes_tensor, hidden_state)
            # don't have node
            else:
                chose_nodes = torch.zeros((1, self.CHOSE_IMPORTENT_NODE_OUTPUT_SHAPE))
                if self.cfg.SCENE_GRAPH.GPU:
                    chose_nodes = chose_nodes.to(self.device)
        else:
            x = x.clone().detach()
            attributes = attributes.clone().detach()
            edge_obj_to_obj = edge_obj_to_obj.clone().detach()
            word_feature = self.node_word_embed_downsample(x[:, :self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE])
            rgb_feature = self.node_rgb_feature_downsample(x[:, self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE:])
            # torch.Size([2, 16])
            x = torch.cat([word_feature, rgb_feature], dim=1)
            x = F.relu(self.dropout(x))
            x = F.relu(self.conv1(x, edge_obj_to_obj, edge_weight))
            x = self.dropout(x)
            x = torch.cat([x, attributes], dim=1)
            # torch.Size([2, 40])
            x = self.conv2(x, edge_obj_to_obj, edge_weight)
            x = F.relu(self.dropout(x))
            # torch.Size([2, 16]) + torch.Size([2, 24]) => torch.Size([2, 40])
            x = torch.cat([x, attributes], dim=1)
            nodes_tensor = x
            # torch.Size([1, 400])
            chose_nodes, dict_ANALYZE_GRAPH = self.chose_node_module(nodes_tensor, hidden_state)
            # print(x.requires_grad)
        mapping_feature = self.chose_node_feature_mapping(chose_nodes)
        return mapping_feature, dict_ANALYZE_GRAPH

    def priori_feature(self, data, hidden_state):
        x, attributes, edge_obj_to_obj, edge_weight = \
            data.x, \
            data.attributes, \
            data.edge_obj_to_obj, \
            data.edge_attr
        dict_ANALYZE_GRAPH = None

        x = x.clone().detach()
        word_feature = self.node_word_embed_downsample(x[:, :self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE])
        rgb_feature = self.node_rgb_feature_downsample(x[:, self.cfg.SCENE_GRAPH.NODE_INPUT_WORD_EMBED_SIZE:])
        # torch.Size([2, 16])
        x = torch.cat([word_feature, rgb_feature], dim=1)
        x = F.relu(self.dropout(x))
        x = F.relu(self.conv1(x, edge_obj_to_obj, edge_weight))
        x = self.dropout(x)
        x = torch.cat([x, attributes], dim=1)
        # torch.Size([2, 40])
        x = self.conv2(x, edge_obj_to_obj, edge_weight)
        x = F.relu(self.dropout(x))
        # torch.Size([2, 16]) + torch.Size([2, 24]) => torch.Size([2, 40])
        x = torch.cat([x, attributes], dim=1)
        nodes_tensor = x
        # torch.Size([1, 400])
        chose_nodes, dict_ANALYZE_GRAPH = self.chose_node_module(nodes_tensor, hidden_state)
        mapping_feature = self.chose_node_feature_mapping(chose_nodes)
        return mapping_feature, dict_ANALYZE_GRAPH
