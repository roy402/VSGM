# SGG training data
https://github.com/jwyang/graph-rcnn.pytorch
```
pip install -r requirements.txt
apt-get update
apt-get install libglib2.0-0
apt-get install libsm6
cd lib/scene_parser/rcnn
python setup.py build develop
```

You have to change $GRAPH_RCNN_ROOT/configs/attribute.yaml parameters
ROI_BOX_HEAD.NUM_CLASSES : len(gen.constants.OBJECTS_DETECTOR) + 1 (background)
ROI_RELATION_HEAD.NUM_CLASSES : 1 (if object has parentReceptacle) + 1 (background) + 2 (Assertion t >= 0 && t < n_classes failed, https://github.com/facebookresearch/maskrcnn-benchmark/pull/1214)
ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES : ? (visible, isToggled ...)

SGG train transformers would .convert("RGB")/255


## 1.preload ALFWORLD images, masks, meta, sgg_meta
/home/alfworld/agents/sgg/alfred_data_format.py
/home/alfworld/agents/sgg/parser_scene.py


## 2.generate images, masks, meta, sgg_meta
```
python gen/scripts/augment_sgg_trajectories.py --data_path data/json_2.1.1/train --save_path detector/data/test
```
bathroom 5257 bedroom 6060 kitchen 41775 living 6331

### meta
```
{"(95, 252, 121)": "Mug|+02.82|+00.79|-01.22", "(8, 94, 186)": "Mug", "(108, 125, 127)": "CD|-01.29|+00.68|-00.85", ...}
```

### sgg_meta
```
{
  "name": "Mug_2a940808(Clone)_copy_28",
  "position": {
    "x": 2.82123375,
    "y": 0.788715661,
    "z": -1.22375751
  },
  "rotation": {
    "x": 0.00006351283,
    "y": 0.000053627562,
    "z": -0.000171551466
  },
  "cameraHorizon": 0,
  "visible": false,
  "receptacle": true,
  "toggleable": false,
  "isToggled": false,
  "breakable": true,
  "isBroken": false,
  "canFillWithLiquid": true,
  "isFilledWithLiquid": false,
  "dirtyable": true,
  "isDirty": false,
  "canBeUsedUp": false,
  "isUsedUp": false,
  "cookable": false,
  "isCooked": false,
  "ObjectTemperature": "RoomTemp",
  "canChangeTempToHot": false,
  "canChangeTempToCold": false,
  "sliceable": false,
  "isSliced": false,
  "openable": false,
  "isOpen": false,
  "pickupable": true,
  "isPickedUp": false,
  "mass": 1,
  "salientMaterials": [
    "Ceramic"
  ],
  "receptacleObjectIds": [],
  "distance": 3.2283268,
  "objectType": "Mug",
  "objectId": "Mug|+02.82|+00.79|-01.22",
  "parentReceptacle": null,
  "parentReceptacles": [
    "Desk|+02.29|+00.01|-01.15"
  ],
  "currentTime": 0,
  "isMoving": true,
  "objectBounds": {
    "objectBoundsCorners": [
      {
        "x": 2.88107681,
        "y": 0.7887154,
        "z": -1.16639853
      },
      {
        "x": 2.736586,
        "y": 0.788715839,
        "z": -1.16639841
      },
      {
        "x": 2.736586,
        "y": 0.788715959,
        "z": -1.28111649
      },
      {
        "x": 2.88107681,
        "y": 0.788715541,
        "z": -1.2811166
      },
      {
        "x": 2.88107729,
        "y": 0.9005291,
        "z": -1.16639841
      },
      {
        "x": 2.73658657,
        "y": 0.9005295,
        "z": -1.16639829
      },
      {
        "x": 2.73658633,
        "y": 0.9005296,
        "z": -1.28111625
      },
      {
        "x": 2.881077,
        "y": 0.9005292,
        "z": -1.28111649
      }
    ]
  }
},
```

## 3.Test
### Check alfred detector data is ok 
$GRAPH_RCNN_ROOT/configs/attribute.yaml
'''
cd /home/alfworld/agents
python sgg/alfred_data_format.py config/test_base.yaml --semantic_config_file config/semantic_graph.yaml --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml

// windows
python sgg/alfred_data_format.py config/test_base.yaml --semantic_config_file config/semant ic_graph.yaml --sgg_config_file sgg/graph-rcnn.pytorch/configs/attribute.yaml
'''

### Check oracle scene graph is ok
'''
cd /home/alfworld/agents
python semantic_graph/semantic_graph.py config/semantic_graph_base.yaml --semantic_config_file config/semantic_graph.yaml
'''


## 4. train "Scene graph generation"-object relation attribute
### 1. Get SGG Train Data
Read $ALFWORLD_ROOT/agents/sgg/TRAIN_SGG.md

### 2. Train SGG
- label 會自動+1 (__background__)，導致輸出label都被多往前了一位
- 因此TransMetaData的ind_to_classes，要比object_classes還要多insert(0, '__background__')，如果想要修改這問題，object_classes不需要insert(0, '__background__')
```
object_classes = ['AlarmClock', 'Apple', 'AppleSliced', ...]
class_to_ind = ['__background__', 'AlarmClock', 'Apple', ...]
```
- train with instance mask
```
AlfredDataset
img_path = self.masks[idx]
```

```
cd $GRAPH_RCNN_ROOT/
CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config-file configs/attribute.yaml
CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/attribute.yaml
```
### 3. Eval SGG result
1. Pretrain model PATH: MODEL.WEIGHT_DET or --resume 124999 (checkpoint_0124999.pth) or MODEL.WEIGHT_IMG
Change $GRAPH_RCNN_ROOT/configs/attribute.yaml parameter
2. --visualize True (--inference 進行測試)
```
--instance (測試產生visualize的照片數量)
MODEL.SAVE_SGG_RESULT_PATH (測試產生visualize的照片儲存位置)
```
```
CUDA_VISIBLE_DEVICES=1 python main.py --config-file configs/attribute.yaml --inference --resume 179999 --visualize --instance 10
# store image in 
        visualize_folder = "visualize"
        if not os.path.exists(visualize_folder):
```

#### CUDA out of memory
reduce PRE_NMS_TOP_N_TEST, POST_NMS_TOP_N_TEST, FPN_POST_NMS_TOP_N_TEST

### 4. SGG - Test model OK on ALFRED & ALFWORLD
```
1. Pretrain model PATH: MODEL.WEIGHT_IMG 
Change $GRAPH_RCNN_ROOT/configs/attribute.yaml parameter
2.
CUDA_VISIBLE_DEVICES=1 python sgg/sgg.py config/semantic_graph_base.yaml --semantic_config_file config/semantic_graph.yaml --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml
```
- SGG result attribute
"labels"
"features"
"obj_relations_idx_pairs"
"obj_relations_scores"
"attribute_logits"

#### ERROR
attribute.yaml setting different with training 
```
load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for SGG:
        Unexpected key(s) in state_dict: "rel_heads.rel_predictor.pred_feature_extractor.head.layer4.0.downsample.0.weight", "rel_heads.rel_predictor.pred_feature_extractor.head.layer4.0.downsample.1.weight", "rel_heads.rel_predictor.pred_feature_extractor.head.layer4.0.downsample.1.bias"
```
Number of proposals 1 is too small, retrying filter_results with score thresh = 0.0015625

### Train with semantic graph dagger
- ANALYZE_GRAPH
```
_C.GENERAL.ANALYZE_GRAPH = True
dict_ANALYZE_GRAPH = {
    "score": score.clone().detach().to('cpu'),
    "sort_nodes_index": sort_nodes_index.clone().detach().to('cpu'),
}
```
