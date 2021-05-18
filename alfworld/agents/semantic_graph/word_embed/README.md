# Word embedding 
For semantic_graph object class name to fasttext word embedding

## Get Object feature *.csv
1. change $GRAPH_ANALYSIS/data/objects.txt to new object class (class object must be same with yaml file "ALFREDTEST.object_types")
2. 
```
cd graph_analysis
python create_dgl_data.py --object
```
You will create csv file at 'graph_analysis/data_dgl/object_alfworld.csv'