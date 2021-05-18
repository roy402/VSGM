import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import nn.vnn as vnn
import math
import scipy.sparse as sp
import numpy as np
import h5py


# SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS
# A^ = D^{-1/2} * A * D^{-1/2}
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class WordEmbeddingLoad:
    def __init__(self, h5py_file):
        self.word_embeddings = h5py.File(h5py_file, "r")


# 83 objects
A_raw = torch.load("./data/gcn/adjmat_83.dat")
# 108 objects
A_raw = np.asarray(np.genfromtxt("./data/gcn/A_108.csv", delimiter=','))
A_raw = torch.tensor(A_raw)
WEh5 = WordEmbeddingLoad("./data/gcn/glove_map300d_83.hdf5")
WEh5 = WordEmbeddingLoad("./data/gcn/fastText_300d_108.h5")


class GCN(nn.Module):
    def __init__(self, noutfeat, nhid=1024, dropout=0.5):
        super(GCN, self).__init__()

        # get and normalize adjacency matrix.
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A)).unsqueeze(0).cuda()

        # word embed for all the objs.
        objects = open("./data/gcn/objects_108.txt").readlines()
        objects = [o.strip() for o in objects]
        self.n = len(objects)
        word_embeddings = torch.zeros(self.n, 300)
        for i in range(self.n):
            word_embeddings[i, :] = torch.Tensor(WEh5.word_embeddings[objects[i]][:])
        x = WEh5.word_embeddings.keys()
        self.word_embeddings = nn.Parameter(word_embeddings.unsqueeze(0))
        self.word_embeddings.requires_grad = False

        # GCN layer
        self.gc1 = GraphConvolution(self.word_embeddings.shape[2], nhid)
        self.gc2 = GraphConvolution(nhid, 1)
        self.dropout = dropout

        self.final_mapping = nn.Linear(self.n, noutfeat)

    def forward(self, frames):
        batch_size = frames.shape[0] * frames.shape[1]
        batch_ex_size = frames.shape[0]
        word_embed = self.word_embeddings.detach()
        x = F.relu(self.gc1(word_embed, self.A))
        x = F.dropout(x, self.dropout, training=self.training)

        # [1, 83, 1024] -> [122, 83, 1024]
        x = x.repeat(batch_size, 1, 1)
        x = F.relu(self.gc2(x, self.A))
        x = x.view(batch_ex_size, -1, self.n)
        # [2, 61, 512]
        x = self.final_mapping(x)
        # import pdb; pdb.set_trace()
        return x


class GCNVisual(GCN):
    def __init__(self, noutfeat, nhid=1024, dropout=0.5):
        super(GCNVisual, self).__init__(noutfeat, nhid, dropout)
        # vis_encoder
        dframe = 128
        self.vis_encoder = vnn.ResnetVisualEncoder(dframe=dframe)
        self.gc2 = GraphConvolution(nhid+dframe, 1)

    def forward(self, frames):
        batch_size = frames.shape[0]
        word_embed = self.word_embeddings.detach()
        # gc1
        x = F.relu(self.gc1(word_embed, self.A))
        x = F.dropout(x, self.dropout, training=self.training)
        # concat frames feats, gc1 x
        # [122, 128]
        encode_frames = self.vis_encoder(frames.view(-1, frames.shape[2], frames.shape[3], frames.shape[4]))
        # [122, 1, 128], [122, 83, 128]
        encode_frames = encode_frames.unsqueeze(1).repeat(1, x.shape[1], 1)

        # [1, 83, 1024] -> [122, 83, 1024]
        x = x.repeat(encode_frames.shape[0], 1, 1)
        x = torch.cat((encode_frames, x), dim=-1)
        # gc2
        x = F.relu(self.gc2(x, self.A))
        x = x.view(batch_size, -1, self.n)
        x = self.final_mapping(x)
        return x


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(1, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight.expand(input.shape[0], -1, -1))
        output = torch.bmm(adj.expand(support.shape[0], -1, -1), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
