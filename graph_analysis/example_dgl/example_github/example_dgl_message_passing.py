import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.
    
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)
    
    def forward(self, g, h):
        """Forward computation
        
        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        # All the `ndata` set within a local scope will be automatically popped out.
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h_total = torch.cat([h, h_neigh], dim=1)
            return F.relu(self.linear(h_total))

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.
    
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)
    
    def forward(self, g, h, w):
        """Forward computation
        
        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        w : Tensor
            The edge weight.
        """
        # All the `ndata` set within a local scope will be automatically popped out.
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = w
            # update_all is a message passing API.
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            h_total = torch.cat([h, h_neigh], dim=1)
            return F.relu(self.linear(h_total))

def u_mul_e_udf(edges):
    return {'m' : edges.src['h'] * edges.data['w']}
