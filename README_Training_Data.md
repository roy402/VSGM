## Gen Data
1. raw_images, sgg_meta, exploration_meta
```
cd alfred/gen
python scripts/augment_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
python scripts/augment_meta_data_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
```

2. extract exploration img to resnet feature to 'feat_third_party_img_and_exploration.pt'
feat_depth_instance.pt
feat_third_party_img_and_exploration.pt
feat_sgg_depth_instance.pt
```
python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 64 --gpu --visual_model resnet18 --img_folder high_res_images,raw_images_1,raw_images_2,exploration_meta,exploration_meta_1,exploration_meta_2 --keyname feat_conv,feat_conv_1,feat_conv_2,feat_exploration_conv,feat_exploration_conv_1,feat_exploration_conv_2 --skip_existing --str_save_ft_name feat_third_party_img_and_exploration.pt

python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 64 --gpu --visual_model resnet18 --img_folder depth_images,instance_masks --keyname depth,instance --skip_existing --str_save_ft_name feat_depth_instance.pt

CUDA_VISIBLE_DEVICES=1 python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 64 --gpu --visual_model sgg --img_folder depth_images,instance_masks --keyname depth,instance --skip_existing --str_save_ft_name feat_sgg_depth_instance.pt --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml --task_types 1,2,3,4,5,6
--task_types 1
```

3. get event.metadata['agent']
cameraHorizon, position, rotation
https://ai2thor.allenai.org/ithor/documentation/metadata/#agent
```
python scripts/augment_agent_meta_data.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```
agent
"position": {"x": -2.25, "y": 0.9009992, "z": 1.25}, "rotation": {"x": 0.0, "y": 180.0, "z": 0.0}
object
"name": "Mug_a12c171b(Clone)copy24", "position": {"x": -0.6516613, "y": 0.8507198, "z": -0.6635949}, "rotation": {"x": 0.00138341438, "y": 0.000116876137, "z": -0.0008958757}

WARNING: if use "feat_sgg_xxx.pt", config "FEAT_NAME", "NODE_INPUT_RGB_FEATURE_SIZE", "PRIORI_OBJ_RBG_FEATURE_EMBEDDING" need change



# Third Party Camera Frames (Not work with ai2thor==2.1.0)
```
python scripts/augment_trajectories_third_party_camera_frames210.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
python scripts/augment_meta_data_trajectories_third_party_camera_frames.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```
merge thirdparty meta to 'third_party_all_meta_data.json' & test feat_third_party_img_and_exploration, feat_depth_instance, feat_sgg_depth_instance is ok
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --data data/full_2.1.0/ --model merge_meta_im --dout exp/just_merge_meta --splits data/splits/oct21.json --batch 20 --gpu
--task_types 1
(windows: python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --model merge_meta_im --dout exp/just_merge_meta --splits data/splits/oct21.json --batch 20 --task_types 1)
```

get_third_party_frame
set_hand_object_hide
https://github.com/allenai/ai2thor/issues/537
### get raw_images_2, raw_images_1
```
python scripts/augment_trajectories_third_party_camera_frames210.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```

### get exploration_meta_1, sgg_meta_1
```
python scripts/augment_meta_data_trajectories_third_party_camera_frames.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```

---
---

## Add Camera (ai2thor==2.5.4)
```
controller.step('AddThirdPartyCamera',
    rotation=dict(x=0, y=90, z=0), 
    position=dict(x=-1.25, y=1.0, z=-1.0),
    fieldOfView=60)
```

```
controller.step('UpdateThirdPartyCamera',
    thirdPartyCameraId=0, # id is available in the metadata response
    rotation=dict(x=0, y=90, z=0),
    position=dict(x=-1.25, y=1.0, z=-1.5)
    )
```

### get Camera frame
```
event.metadata['thirdPartyCameras']
```
{'thirdPartyCameraId': 0, 'position': {'x': -0.75, 'y': 0.9101201295852661, 'z': 0.25}, 'rotation': {'x': 0.0, 'y': 270.0, 'z': 0.0}, 'fieldOfView': 90.0}
{'thirdPartyCameraId': 1, 'position': {'x': -0.75, 'y': 0.9101201295852661, 'z': 0.25}, 'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0}, 'fieldOfView': 90.0}
```
event.third_party_camera_frames
# len(event.third_party_camera_frames) = 2
```

### metadata['agent']
event.metadata['agent']
```
{{'name': 'agent', 'position': {'x': -0.75, 'y': 0.9101201295852661, 'z': 0.25}, 'rotation': {'x': -0.0, 'y': 0.0, 'z': 0.0}, 'cameraHorizon': 30.000003814697266, 'isStanding': True, 'inHighFrictionArea': False}
```
