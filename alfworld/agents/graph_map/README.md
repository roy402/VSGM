# Graph map

WARNING: depth image can't be transforms
## Test graph_map
It will generate gif at $ALFWORLD_ROOT/agents/
```
python graph_map/graph_map.py
```

1.
BasicGraphMap
  map
  	map: 3 dim, self.S, self.S, self.CLASSES

2.
GraphMap
  map
	x: 1 dim
	attributes: 1 dim
	# self.map.activate_nodes = set()
	activate_nodes: 1 dim
	queue_grid_layer: 1 dim
one_dim_coor = x + self.S * z + self.S * self.S * labels

3.
GraphMapV2
  map
	# self.map.activate_nodes = dict()
  	activate_nodes: {5: 1, ...} node_index & label
visualize_graph_map()
imageio.get_writer('./graph_map.gif', mode='I')

## Test slam_map
It will generate image at $ALFWORLD_ROOT/agents/slam_dump
'''
python graph_map\slam_map.py
'''