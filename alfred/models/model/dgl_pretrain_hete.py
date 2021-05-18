import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from tensorboardX import SummaryWriter
try:
    import model.dgl_gcn_hete as dgl_gcn_hete
except Exception as e:
    import dgl_gcn_hete

# SET LABEL
SET_OBJECT_LABEL = 0
SET_LOWACTION_LABEL = 1
SET_SUBGOAL_LABEL = 2

# import pdb; pdb.set_trace()


class THETLOWSG(dgl_gcn_hete.HETLOWSG):
    """docstring for ClassName"""

    def __init__(self, args, HETAttention, o_feats_dgcn, device="cpu"):
        super().__init__(HETAttention, o_feats_dgcn, device)
        # self.opt = torch.optim.Adam(self.parameters(), lr=args.lr)
        n_node_label, n_edge_label = self._graph_label_create(self.g)
        self.pred_nodes = nn.Linear(o_feats_dgcn, n_node_label)
        self.pred_edges = MLPPredictor(o_feats_dgcn, n_edge_label)
        self.train_iter = 0

    def _graph_label_create(self, g):
        print("=== graph_label_create ===")
        # ('object', 'interacts', 'object'), ('room', 'interacts', 'object'), ('attribute', 'behave', 'object'), ('lowaction', 'belone', 'subgoal'), ('subgoal', 'belone', 'lowaction')
        list_predict_node = ["object", "lowaction", "subgoal"]
        list_node_label = [SET_OBJECT_LABEL, SET_LOWACTION_LABEL, SET_SUBGOAL_LABEL]
        n_node_label = len(list_predict_node)
        n_edge_label = len(g.canonical_etypes)
        for ind_label, c_etype in enumerate(g.canonical_etypes):
            srctype, etype, dsttype = c_etype
            print(c_etype)
            # edges predict, edges input "all", "uv" or "eid"
            num_edge = len(g.edges('eid', etype=c_etype))
            labels = torch.empty(num_edge, dtype=torch.long, device=self.device).fill_(ind_label)
            train_mask = torch.zeros(num_edge, dtype=torch.long, device=self.device).bernoulli(0.6)
            g.edges[c_etype].data['label'] = labels
            g.edges[c_etype].data['train_mask'] = train_mask
        # node predict
        for node, set_label in zip(list_predict_node, list_node_label):
            num_node = g.nodes(node).shape
            labels = torch.empty(num_node, dtype=torch.long, device=self.device).fill_(set_label)
            train_mask = torch.zeros(num_node, dtype=torch.long, device=self.device).bernoulli(0.6)
            g.nodes[node].data['label'] = labels
            g.nodes[node].data['train_mask'] = train_mask
        return n_node_label, n_edge_label

    def _get_h(self):
        in_feat = {"object": self.g.nodes["object"].data['feature'],
                   "room": self.g.nodes['room'].data['feature'],
                   "attribute": self.g.nodes['attribute'].data['feature'],
                   }
        h = self.conv1(self.g, in_feat)
        h = {"object": h["object"],
             "subgoal": self.g.nodes['subgoal'].data['feature'],
             "lowaction": self.g.nodes['lowaction'].data['feature'],
             }
        h = self.conv2(self.g, h)
        return h

    def predict_nodes(self, ):
        h = self._get_h()
        dict_logits = {}
        for k, v in h.items():
            dict_logits[k] = self.pred_nodes(v)
        # import pdb;pdb.set_trace()
        return dict_logits

    def predict_edges(self, ):
        h = self._get_h()
        # import pdb;pdb.set_trace()
        return self.pred_edges(self.g, h)

    # https://docs.dgl.ai/guide/training-node.html
    def train_nodes(self, opt, summary_writer):
        self.train()
        # forward propagation by using all nodes
        total_loss = 0
        dict_logits = self.predict_nodes()
        for k, logits in dict_logits.items():
            node_labels = self.g.nodes[k].data['label']
            train_mask = self.g.nodes[k].data['train_mask']
            # compute loss
            loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
            total_loss += loss
        # compute validation accuracy
        # import pdb;pdb.set_trace()
        acc = self.evaluate()
        # backward propagation
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        summary_writer.add_scalar('graph/heteG_node_loss', total_loss.item(), self.train_iter)
        summary_writer.add_scalar('graph/heteG_node_acc', acc, self.train_iter)
        self.train_iter += 1

    def evaluate(self):
        self.eval()
        with torch.no_grad():
            acc = []
            dict_logits = self.predict_nodes()
            for k, logits in dict_logits.items():
                node_labels = self.g.nodes[k].data['label']
                train_mask = self.g.nodes[k].data['train_mask']
                logits = logits[train_mask]
                labels = node_labels[train_mask]
                _, indices = torch.max(logits, dim=1)
                correct = torch.sum(indices == labels)
                acc.append(correct.item() * 1.0 / len(labels))
            return sum(acc)/len(acc)


# https://docs.dgl.ai/guide/training-edge.html
class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score']


if __name__ == '__main__':
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
    from tensorboardX import SummaryWriter
    import sys
    import os
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models', 'model'))
    import dgl_gcn_hete
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--dout', help='optimizer learning rate', default=".")
    args = parser.parse_args()
    # parse arguments
    test_mode = THETLOWSG(args, True, 300)
    summary_writer = SummaryWriter(log_dir="./exp/test/")
    # pretrain graph
    for i in range(200):
        test_mode.train_nodes(summary_writer)
