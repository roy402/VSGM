import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp

from tutorial_utils import load_zachery

# ----------- 0. load graph -------------- #
g = load_zachery()
print(g)

# ----------- 1. node features -------------- #
node_embed = nn.Embedding(g.number_of_nodes(), 5)  # Every node has an embedding of size 5.
inputs = node_embed.weight                         # Use the embedding weight as the node features.
nn.init.xavier_uniform_(inputs)

# Split edge set for training and testing
u, v = g.edges()
eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)


test_pos_u, test_pos_v = u[eids[:50]], v[eids[:50]]
train_pos_u, train_pos_v = u[eids[50:]], v[eids[50:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(34)
neg_u, neg_v = np.where(adj_neg != 0)
neg_eids = np.random.choice(len(neg_u), 200)
test_neg_u, test_neg_v = neg_u[neg_eids[:50]], neg_v[neg_eids[:50]]
train_neg_u, train_neg_v = neg_u[neg_eids[50:]], neg_v[neg_eids[50:]]

# Create training set.
train_u = torch.cat([torch.as_tensor(train_pos_u), torch.as_tensor(train_neg_u)])
train_v = torch.cat([torch.as_tensor(train_pos_v), torch.as_tensor(train_neg_v)])
train_label = torch.cat([torch.zeros(len(train_pos_u)), torch.ones(len(train_neg_u))])

# Create testing set.
test_u = torch.cat([torch.as_tensor(test_pos_u), torch.as_tensor(test_neg_u)])
test_v = torch.cat([torch.as_tensor(test_pos_v), torch.as_tensor(test_neg_v)])
test_label = torch.cat([torch.zeros(len(test_pos_u)), torch.ones(len(test_neg_u))])

from dgl.nn import SAGEConv

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
# Create the model with given dimensions 
# input layer dimension: 5, node embeddings
# hidden layer dimension: 16
net = GraphSAGE(5, 16)

# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), node_embed.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    logits = net(g, inputs)
    pred = torch.sigmoid((logits[train_u] * logits[train_v]).sum(dim=1))
    import pdb; pdb.set_trace()
    
    # compute loss
    loss = F.binary_cross_entropy(pred, train_label)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_logits.append(logits.detach())
    
    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
pred = torch.sigmoid((logits[test_u] * logits[test_v]).sum(dim=1))
print('Accuracy', ((pred >= 0.5) == test_label).sum().item() / len(pred))