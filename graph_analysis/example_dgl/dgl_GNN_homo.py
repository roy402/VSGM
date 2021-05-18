import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.nn import SAGEConv
import numpy as np
import pandas as pd
# import nn.vnn as vnn


# import pdb; pdb.set_trace()
def load_object_interact_object():
    nodes_data = pd.read_csv('../data_dgl/object.csv')
    edges_data = pd.read_csv('../data_dgl/object-interact-object.csv')
    src = edges_data['Src'].to_numpy()
    dst = edges_data['Dst'].to_numpy()
    g = dgl.graph((src, dst))
    Ids = nodes_data['Id'].to_list()
    Ids = torch.tensor(Ids).long()
    # We can also convert it to one-hot encoding.
    feature = [nodes_data['feature'].to_list()]
    for i in range(1, 300):
        feature.extend([nodes_data['feature.{}'.format(i)].to_list()])
    feature = torch.tensor(feature).float().transpose(0, 1)
    g.ndata.update({'Ids': Ids, 'feature': feature})

    import pdb; pdb.set_trace()
    g.edges()
    return g


g = load_object_interact_object()
inputs = g.ndata['feature']


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, num_classes, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


net = GraphSAGE(300, 16, 2)

for e in range(100):
    # forward
    logits = net(g, inputs)
    import pdb; pdb.set_trace()
