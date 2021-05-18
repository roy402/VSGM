# ALFWorld
## requirements
https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
download whl handly and install
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
pip install torch-geometric
```

## Word embedding 
For semantic_graph object class name to fasttext word embedding

### Get Object word feature '.csv'
change graph_analysis/data/objects.txt
```
cd graph_analysis
python create_dgl_data.py --object
```
You will create csv file at 'graph_analysis/data_dgl/object_alfworld.csv'

## Get img_feature feature data
Please see $ALFWORLD_ROOT/agents/sgg/TRAIN_SGG.md "2.generate images, masks, meta, sgg_meta".
Gen detector train data first. For AlfredDataset class load data

1. rgb_feature
set SAVE_PATH, OBJECT_RGB_FEATURE = rgb()
```
python sgg/alfred_data_format.py config/test_base.yaml --semantic_config_file config/semantic_graph.yaml --not_save_config
```
2. mask_feature
set SAVE_PATH, OBJECT_RGB_FEATURE = mask()
```
python sgg/alfred_data_format.py config/test_base.yaml --semantic_config_file config/semantic_graph.yaml --not_save_config
```
3. sgg_mask_feature
set SAVE_PATH, OBJECT_RGB_FEATURE = sgg_mask()
```
python sgg/alfred_data_format.py config/test_base.yaml --semantic_config_file config/semantic_graph.yaml --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml --not_save_config
```
4. sgg_mask_2048
set SAVE_PATH, OBJECT_RGB_FEATURE = sgg_mask_2048()

## semantic_graph
### GraphData & HeteGraphData
node: [word 300 + rgb feature 512] or [word 300 + rgb feature 2048]
attribute: 23 + 1
relation: only have relation "0".
```
# update_relation didn't use relation para
def update_relation(self, node_src, node_dst, relation):
```
relation sgg: "0" no relation. "1" ai2-thor relation

## Test Memory Graph is OK
```
python semantic_graph/semantic_graph.py config/semantic_graph_base.yaml --semantic_config_file config/semantic_graph.yaml --not_save_config
```

- obj_cls_name_to_features
	- obj_cls_name would +1
	- cause
		- '__background__' = 0