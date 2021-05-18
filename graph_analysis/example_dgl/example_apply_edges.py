import dgl
import torch

g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
g.ndata['h'] = torch.ones(5, 100)
g.apply_edges(lambda edges: {'x' : edges.src['h'] + edges.dst['h']})
g.edata['x']


import dgl.function as fn
g.apply_edges(fn.u_add_v('h', 'h', 'x'))
g.edata['x']
import pdb; pdb.set_trace()
g.apply_edges(fn.copy_src(src='h', out='score'))
g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
g.edata['x']

g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1])})
g.edges[('user', 'plays', 'game')].data['h'] = torch.ones(4, 5)
g.apply_edges(lambda edges: {'h': edges.data['h'] * 2})
g.edges[('user', 'plays', 'game')].data['h']