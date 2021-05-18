import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
import graph_embed
import numpy as np

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
            self.CHOSE_IMPORTENT_NODE_OUTPUT_SHAPE = self.EMBED_FEATURE_SIZE
            self.chose_node_module = graph_embed.WeightedSoftmaxSum(
                bert_hidden_size,
                NODE_FEATURE_SIZE,
                self.EMBED_FEATURE_SIZE,
                NUM_CHOSE_NODE,
                cfg.SCENE_GRAPH.GPU,
                PRINT_DEBUG=PRINT_DEBUG,
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
            # don't have node
            else:
                x = torch.zeros((1, self.EMBED_FEATURE_SIZE))
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
        return chose_nodes, dict_ANALYZE_GRAPH

    def chose_importent_node_graph_map(self, data, hidden_state):
        x, attributes, edge_obj_to_obj, edge_weight = \
            data.x, \
            data.attributes, \
            data.edge_obj_to_obj, \
            data.edge_attr
        edge_obj_to_obj, edge_weight =\
            subgraph(data.activate_nodes, edge_obj_to_obj, edge_attr=edge_weight)
        # copy from chose_importent_node
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
        return chose_nodes, dict_ANALYZE_GRAPH


# https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/utils/subgraph.py
def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False,
             num_nodes=None):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.
    Args:
        subset (LongTensor, BoolTensor or [int]): The nodes to keep.
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """

    device = edge_index.device
    #  0: Sizes of tensors must match except in dimension 1. Got 2 and 43444 in dimension 0
    if isinstance(subset, set):
        subset = list(subset)
    # GraphMapV2
    if isinstance(subset, dict):
        subset = list(subset.keys())
    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, torch.Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))