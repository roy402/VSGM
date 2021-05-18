import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl.function as fn
import dgl.nn.pytorch as dglnn
# import nn.vnn as vnn


# import pdb; pdb.set_trace()
def load_heterograph():
    # edges
    edges_data = pd.read_csv('../data_dgl/object-interact-object.csv')
    src_objectobject = edges_data['Src'].to_numpy()
    dst_objectobject = edges_data['Dst'].to_numpy()
    edges_data = pd.read_csv('../data_dgl/room-interact-object.csv')
    src_roomobject = edges_data['Src'].to_numpy()
    dst_roomobject = edges_data['Dst'].to_numpy()
    edges_data = pd.read_csv('../data_dgl/attribute-behave-object.csv')
    src_attributeobject = edges_data['Src'].to_numpy()
    dst_attributeobject = edges_data['Dst'].to_numpy()
    graph_data = {
        ('object', 'interacts', 'object'): (src_objectobject, dst_objectobject),
        ('room', 'interacts', 'object'): (src_roomobject, dst_roomobject),
        ('attribute', 'behave', 'object'): (src_attributeobject, dst_attributeobject),
    }
    # graph
    g = dgl.heterograph(graph_data)
    # nodes
    csv_nodes_object = pd.read_csv('../data_dgl/object.csv')
    csv_nodes_attribute = pd.read_csv('../data_dgl/attribute.csv')
    csv_nodes_room = pd.read_csv('../data_dgl/room.csv')
    g.nodes['object'].data['feature'] = get_feature(csv_nodes_object)
    g.nodes['attribute'].data['feature'] = get_feature(csv_nodes_attribute)
    g.nodes['room'].data['feature'] = get_feature(csv_nodes_room)
    return g


def get_feature(csv_nodes_data):
    feature = [csv_nodes_data['feature'].to_list()]
    for i in range(1, 300):
        feature.extend([csv_nodes_data['feature.{}'.format(i)].to_list()])
    feature = torch.tensor(feature).float().transpose(0, 1)
    return feature


# https://docs.dgl.ai/en/latest/api/python/nn.pytorch.html#dgl.nn.pytorch.HeteroGraphConv
# https://docs.dgl.ai/en/latest/guide/nn-heterograph.html
class NetGCN(nn.Module):
    def __init__(self, in_feats, h_feats, o_feats):
        super(NetGCN, self).__init__()
        # activation = F.relu()
        activation = nn.ReLU()
        self.conv1 = dglnn.HeteroGraphConv({
            'interacts': dglnn.GraphConv(in_feats, h_feats, activation=activation),
            'behave': dglnn.GraphConv(in_feats, h_feats, activation=activation)},
            aggregate='sum'
        )
        self.conv2 = dglnn.HeteroGraphConv({
            'interacts': dglnn.GraphConv(h_feats, o_feats)},
            aggregate='sum'
        )

    def forward(self, g):
        in_feat = g.ndata['feature']
        h = self.conv1(g, in_feat)
        h = self.conv2(g, h)
        return h


if __name__ == '__main__':
    g = load_heterograph()
    # inputs = g.nodes['object'].data['feature']
    inputs = g.ndata['feature']
    print(g.num_nodes["object"])
    print(g.ntypes)
    print(g.etypes)
    print(g.canonical_etypes)
    for stype, etype, dtype in g.canonical_etypes:
        print([stype, etype, dtype])
    print(g.edges["behave"])
    import pdb; pdb.set_trace()

    net = NetGCN(300, 16, 5)

    for e in range(100):
        # forward
        logits = net(g)
        print(logits["object"].shpe)
