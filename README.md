# Prerequire
```
conda install -c menpo opencv
# https://github.com/dmlc/dgl/tree/master
conda install -c dglteam dgl-cuda9.0
pip install rdflib==4.2.2
pip install fasttext
pip install pytorch-nlp
pip install imageio
pip install icecream
pip install imageio-ffmpeg
pip install adjustText
from icecream import ic
ic(1)

```


# Pre setting

## ALFRED_ROOT
```
cd gcn/alfred
# linux
export ALFRED_ROOT=/home/alfred/
export ALFWORLD_ROOT=/home/alfworld/
export GRAPH_ANALYSIS=/home/graph_analysis/
export GRAPH_RCNN_ROOT=/home/graph-rcnn.pytorch/

# windows
SET ALFRED_ROOT=D:\alfred\alfred
SET ALFWORLD_ROOT=D:\alfred\alfworld
SET GRAPH_ANALYSIS=D:\alfred\graph_analysis
SET GRAPH_RCNN_ROOT=D:\alfred\alfworld\agents\sgg\graph-rcnn.pytorch
```
```
cd alfred
python3 scripts/startx.py
```


## gen data
1. raw_images, sgg_meta, exploration_meta
```
cd alfred/gen
python scripts/augment_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
python scripts/augment_meta_data_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
```
2. third_party_camera_frames: README_thirdparty_view.md
```
python scripts/augment_trajectories_third_party_camera_frames210.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
python scripts/augment_meta_data_trajectories_third_party_camera_frames.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```
3. extract exploration img to resnet feature to 'feat_third_party_img_and_exploration.pt'
feat_depth_instance.pt
feat_third_party_img_and_exploration.pt
feat_sgg_depth_instance.pt
```
python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 64 --gpu --visual_model resnet18 --img_folder high_res_images,raw_images_1,raw_images_2,exploration_meta,exploration_meta_1,exploration_meta_2 --keyname feat_conv,feat_conv_1,feat_conv_2,feat_exploration_conv,feat_exploration_conv_1,feat_exploration_conv_2 --skip_existing --str_save_ft_name feat_third_party_img_and_exploration.pt

python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 64 --gpu --visual_model resnet18 --img_folder depth_images,instance_masks --keyname depth,instance --skip_existing --str_save_ft_name feat_depth_instance.pt

CUDA_VISIBLE_DEVICES=1 python models/utils/extract_resnet.py --data data/full_2.1.0 --batch 64 --gpu --visual_model sgg --img_folder depth_images,instance_masks --keyname depth,instance --skip_existing --str_save_ft_name feat_sgg_depth_instance.pt --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml --task_types 1,2,3,4,5,6
--task_types 1
```
4. get event.metadata['agent']
cameraHorizon, position, rotation
https://ai2thor.allenai.org/ithor/documentation/metadata/#agent
```
python scripts/augment_agent_meta_data.py --data_path ../data/full_2.1.0/ --num_threads 10 --smooth_nav --time_delays
```
agent
"position": {"x": -2.25, "y": 0.9009992, "z": 1.25}, "rotation": {"x": 0.0, "y": 180.0, "z": 0.0}
object
"name": "Mug_a12c171b(Clone)copy24", "position": {"x": -0.6516613, "y": 0.8507198, "z": -0.6635949}, "rotation": {"x": 0.00138341438, "y": 0.000116876137, "z": -0.0008958757}

5. merge thirdparty meta to 'third_party_all_meta_data.json' & test feat_third_party_img_and_exploration, feat_depth_instance, feat_sgg_depth_instance is ok
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --data data/full_2.1.0/ --model merge_meta_im --dout exp/just_merge_meta --splits data/splits/oct21.json --batch 20 --gpu
--task_types 1
(windows: python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --model merge_meta_im --dout exp/just_merge_meta --splits data/splits/oct21.json --batch 20 --task_types 1)
```

WARNING: if use "feat_sgg_xxx.pt", config "FEAT_NAME", "NODE_INPUT_RGB_FEATURE_SIZE", "PRIORI_OBJ_RBG_FEATURE_EMBEDDING" need change

## Config SETTING
ATTRIBUTE_FEATURE_SIZE: 26   # 23 + 2 (ANGLE_OF_VIEWS) + 1 (unique_obj_index)

## MOCA pre-download eval maskrcnn model
```
cd $ALFRED_ROOT
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
```

## Scene graph generation
Read $ALFWORLD_ROOT/agents/sgg/TRAIN_SGG.md
Read $GRAPH_RCNN_ROOT/README.md
self.config['general']['model']['block_hidden_dim'] = 2 x args.dhid
graph_embed_model -> bert_hidden_size = config['general']['model']['block_hidden_dim']

# Run Model

## MOCA

```
export ALFRED_ROOT=/home/moca/
cd /home/moca
CUDA_VISIBLE_DEVICES=1 python models/train/train_seq2seq.py --model seq2seq_im_mask --dout exp/moca_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu
# eval
python models/eval/eval_seq2seq.py --model models.model.seq2seq_im_mask --model_path exp/moca_seq2seq_im_mask,name,pm_and_subgoals_01/best_seen.pth --eval_split valid_seen --gpu
--subgoals GotoLocation
```


## MOCA + Semantic
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_semantic --dout exp/moca_memory{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 20 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu

CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/memory_semantic_graph.yaml --model_path exp/moca_memoryseq2seq_im_moca_semantic,name,pm_and_subgoals_01_12-01-2021_03-37-50/best_seen.pth --model seq2seq_im_moca_semantic --data data/full_2.1.0/ --eval_split train --gpu
```

## MOCA + Importent nodes & rgb node feature
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/importent_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/importent_rgb_nodes_DynamicNode_moca --splits data/splits/oct21.json --batch 20 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --demb 100 --dhid 256 --gpu

CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/importent_semantic_graph.yaml --model_path exp/importent_nodes_moca_16-01-2021_11-02-31/best_seen.pth --model seq2seq_im_moca_importent_nodes --data data/full_2.1.0/ --eval_split train --gpu
```

## MOCA + Priori
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/priori_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/priori_moca --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu --not_save_config

```

## MOCA + Priori + Graph attention
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/gan_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/graph_attention --splits data/splits/oct21.json --batch 8 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu
```

## Mini MOCA v1-v6
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_graph_softmaxgcn_v{}.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/mini_moca_graph_softmaxgcn_v{} --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1

CUDA_VISIBLE_DEVICES=1 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/importent_semantic_graph_priori_hete_gcn.yaml --model_path exp/mini_moca_graph_softmaxgcn_v3_10-02-2021_16-51-32/best_seen.pth --model seq2seq_im_moca_importent_nodes --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
```

## Moca instance & depth
mini_moca_test_mask_v1.yaml / mini_moca_test_mask_depth_v1.yaml
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_mask_v1.yaml --data data/full_2.1.0/ --model   --dout exp/mini_test_mask_v1 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1

1. (thirdparty have eval bug) 
CUDA_VISIBLE_DEVICES=1 python models/eval_thirdparty/eval_semantic.py models/config/without_env_base.yaml --model_path exp/mini_moca_graph_softmaxgcn_v3_10-02-2021_16-51-32/best_seen.pth --model seq2seq_im_moca_mini_test --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
2.
CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/mini_test_mask_v1_12-02-2021_17-01-40/best_seen.pth --model seq2seq_im_moca_mini_test --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
```

## Moca instance & depth with graph
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_mask_depth_graph_v{}.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_mini_mask_depth --dout exp/mini_moca_test_mask_depth_graph_v{} --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1 

CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/mini_moca_test_mask_depth_graph_v1_12-02-2021_16-59-30/best_seen.pth --model seq2seq_im_moca_mini_mask_depth --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
```

## MOCA + cell_graph (seq2seq_im_moca_mini_mask_depth_big_change)
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_BigChange_mask_graph_v1.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_mini_mask_depth_big_change --dout exp/BigChange_mask_graph_v1 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1

CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/BigChange_mask_graph_v1_21-02-2021_09-47-16/best_seen.pth --model seq2seq_im_moca_mini_mask_depth_big_change --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
```

## sgg feat.pt
1.
--dframe 3*9*9 (origin: 3*7*7)

2.
dframe_channel = 1024 if dframe//2 == 3*9*9 else 512
self.dynamic_conv = DynamicConvLayer(dhid=dhid, d_out_hid=dframe_channel)

3. 
if use "feat_sgg_xxx.pt", config "FEAT_NAME", "NODE_INPUT_RGB_FEATURE_SIZE", "PRIORI_OBJ_RBG_FEATURE_EMBEDDING" need change
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_mask_depth_graph_sgg_feat_v5.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_mini_mask_depth --dout exp/sgg_feat_mini_moca_test_mask_depth_graph_v5 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1 --dframe 243
```

4. eval
```
--sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml
```

## SGG model without Oracle
WARNING: When you eval, must transforms image
```
# eval.py
feat["frame_instance"] = [[frames_instance]]
# Module.step()
self.semantic_graph_implement.trans_MetaData.transforms(feat["frames_instance"])

SGG:
  GPU: 1
```

```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/sgg_without_oracle.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_sgg --dout exp/sgg_without_oracle --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1 --dframe 243 --sgg_pool 2 --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml
```

---
---
# ThirdParty

## Test training data is ok
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/fast_epoch_base.yaml --data data/full_2.1.0/ --model merge_meta_im --dout exp/merge_meta_im --splits data/splits/oct21.json --batch 20 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu
--task_types 1
```

## ThirdParty
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/importent_semantic_graph_softmax_gcn_dec.yaml --data data/full_2.1.0/ --model seq2seq_im_thirdpartyview --dout exp/thirdpartyview_softmax --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1
```
## eval sub goal
```
CUDA_VISIBLE_DEVICES=0 python models/eval_thirdparty/eval_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/importent_semantic_graph_softmax_gcn_dec.yaml --model_path exp/thirdpartyview_softmax/best_seen.pth --model seq2seq_im_thirdpartyview --data data/full_2.1.0/ --eval_split valid_seen --gpu --subgoals all --task_types 1
```

---
---

# Decompose
## define
action_navi_low = ['<<pad>>', '<<seg>>', '<<stop>>', 'LookDown_15', 'MoveAhead_25', 'RotateLeft_90', 'LookUp_15', 'RotateRight_90']
action_operation_low = ['PickupObject', 'SliceObject', 'OpenObject', 'PutObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff']

1.index_to_word => [0-X]
action_navi_low_dict_word_to_index => {k:0, k1:1 ...}
action_operation_low_dict_word_to_index => {k:8, k1:8+1 ...}

2.old_action_low_index_to_navi_or_operation: redefine embed value for train
out_action_navi_or_operation: navi = 0, operation = 1,
ignore_index=0 for 'action_navi_low', 'action_operation_low' ...
ignore_index=-100 for 'action_navi_or_operation'

3.
```
feat['action_low'] # new action low index. for training data (gold)
feat['action_navi_low'] # for training data loss
feat['action_operation_low'] # for training data loss
feat['action_navi_or_operation'] # for training data loss
F.cross_entropy(, reduction='none', ignore_index=0)
```

## Decompose + Adj Relation + Priori + MOCA 
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/decompose_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_decomposed --dout exp/adj_relation_decomposed_priori --splits data/splits/oct21.json --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --action_navi_loss_wt 0.8 --action_oper_loss_wt 1 --action_navi_or_oper_loss_wt 1 --mask_loss_wt 1 --mask_label_loss_wt 1
# eval
CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/decompose_semantic_graph.yaml --model_path exp/fast_epoch_adj_relation_decomposed_priori_24-01-2021_05-45-58/best_seen.pth --model seq2seq_im_decomposed --data data/full_2.1.0/ --eval_split train --gpu
```
'action_low', 'action_navi_low', 'action_operation_low', 'action_navi_or_operation' need pad value = 0

## Decompose2 + fast train
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/decompose_semantic_graph2.yaml --data data/full_2.1.0/ --model seq2seq_im_decomposed --dout exp/fast_decomposed2 --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --action_navi_loss_wt 0.8 --action_oper_loss_wt 1 --action_navi_or_oper_loss_wt 1 --mask_loss_wt 1 --mask_label_loss_wt 1
```

## eval sub goal
```
CUDA_VISIBLE_DEVICES=0 python models/eval_moca/eval_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/decompose_semantic_graph.yaml --model_path exp/fast_decomposed2_26-01-2021_12-22-10/best_seen.pth --model seq2seq_im_decomposed --data data/full_2.1.0/ --eval_split train --gpu --subgoals GotoLocation,PickupObject
```

---
---
---

# Alfred baseline

## Seq2Seq
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model seq2seq_im_mask --dout exp/model,{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
```

## GCN
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1
python models/train/train_seq2seq.py --data data/json_feat_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 
```
--data/json_feat_2.1.0/

gcn visaul embedding
```
cd alfred
python models/train/train_seq2seq.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --gcn_cat_visaul --gpu_id 1
```

## Heterograph GCN
```
python models/train/train_graph.py --data data/full_2.1.0/ --model gcn_im --dout exp/model,heterograph_{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph
python models/train/train_graph.py --data data/json_feat_2.1.0/ --model gcn_im --dout exp/model,heterograph_{model},name,pm_and_subgoals_01,gcn_vial_{gcn_cat_visaul} --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph
```

### depth image
- You must generate depth image first

#### Gen depth image
```
cd alfred/gen
python scripts/augment_trajectories.py --data_path ../data/full_2.1.0/ --num_threads 4 --smooth_nav --time_delays
```
#### check current generate state
```
python scripts/check_augment_state.py --data_path ../data/full_2.1.0/
```
#### Run model
```
cd alfred
python models/train/train_graph.py --data data/full_2.1.0/ --model gcn_depth_im --dout exp/model,heterograph_depth_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --gpu_id 1
```
### HetG + depth + graph attention
```
python models/train/train_graph.py --data data/full_2.1.0/ --model gcn_depth_im --dout exp/model,heterograph_depth_attention_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 4 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg
# eval
python models/eval/eval_graph.py --model_path exp/model,heterograph_oratls_depth_attention_gcn_depth_im,name,pm_and_subgoals_01_19-10-2020_04-46-24/latest.pth --model models.model.gcn_depth_im --data data/full_2.1.0/ --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg --eval_split train
```

### HetG + graph attention
```
python models/train/train_graph.py --data data/full_2.1.0/ --model HeteG_noDepth_im --dout exp/model,heterograph_noDepth_attention_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg --gpu_id 0
# eval
python models/eval/eval_graph.py --model_path exp/model,heterograph_noDepth_attention_HeteG_noDepth_im,name,pm_and_subgoals_01_21-10-2020_05-24-02/latest.pth --model models.model.HeteG_noDepth_im --data data/full_2.1.0/ --model_hete_graph --HETAttention --dgcnout 1024 --HetLowSg --eval_split train
```

### HetG + graph attention + fasttext
```
python models/train/train_graph.py --data data/full_2.1.0/ --model fast_embedding_im --dout exp/model,heterograph__attention_{model},name,pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 1024 --demb 300 --HetLowSg --gpu_id 0
```

## Contrastive
- data
	- [[p1, p2, c4], [c0, h3, h1] ...]
- batch = 2
	- [p1, p2, c4, c0, h3, h1]

### Pretain HetG + graph attention + fasttext + contrastive
```
python models/train/train_parallel.py --data data/full_2.1.0/ --model contrastive_pretrain_im --dout exp/model,SimCLR_{model},heterograph__attention_,name,pm_and_subgoals_01 --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --model_hete_graph --HETAttention --dgcnout 128 --demb 300 --dframe 1000 --dhid 64 --HetLowSg --gpu --gpu_id 0 --DataParallelDevice 0 --DataParallelDevice 1
```

# Eval
## Build AI2-THOR method
1. install https://github.com/allenai/ai2thor-docker
2. 		
(alfred: https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a)
	- ...
	- user console run the X server
```
python3 scripts/startx.py
```

### result
XSERVTransSocketUNIXCreateListener: ...SocketCreateListener() failed
XSERVTransMakeAllCOTSServerListeners: server already running
(have running in docker)

## eval for train/valid_seen/valid_unseen
```
cd alfred
python models/eval/eval_seq2seq.py --model_path exp/json_feat_2/best_seen.pth --model models.model.seq2seq_im_mask --data data/json_feat_2.1.0 --gpu --gpu_id 0
python models/eval/eval_seq2seq.py --model_path exp/model,gcn_im,name,pm_and_subgoals_01,gcn_vial_False_19-09-2020_03-57-55/best_seen.pth --model models.model.gcn_im --data data/json_feat_2.1.0/ --gpu --gpu_id 0 --subgoals all
```
--eval_split ['train', 'valid_seen', 'valid_unseen', ]
--subgoals ['all', 'GotoLocation', 'PickupObject', ...]

## eval hete graph
```
python models/eval/eval_graph.py --model_path exp/{model_path}}/best_seen.pth --model models.model.gcn_im --data data/full_2.1.0/ --gpu --gpu_id 0 --model_hete_graph
```
--HETAttention --dgcnout 1024 --HetLowSg
--eval_split ['train', 'valid_seen', 'valid_unseen', ]
--subgoals ['all', 'GotoLocation', 'PickupObject', ...]
## Leaderboard
```
cd alfred
python models/eval/leaderboard.py --model_path <model_path>/model.pth --model models.model.seq2seq_im_mask --data data/full_2.1.0/ --gpu --num_threads 5
```

---
---

# Build Heterogeneous Graph

## fastText Word Embedding
- https://github.com/facebookresearch/fastText

### Download model
https://fasttext.cc/docs/en/english-vectors.html

### Create Alfred Objects Embedding (GCN Node feature)
You can get "./data/fastText_300d_108.json" & "./data/fastText_300d_108.h5"
```
cd graph_analysis
# download fastText English model & unzip data move to ./data/
./download_model.py en
# use fastText English model get word embedding
python alfred_dataset_vocab_analysis.py
```

## Visaul Genome dataset(GCN Edge Adjacency matrix)
https://visualgenome.org/
You have to download "scene graphs" data first from https://visualgenome.org/api/v0/api_home.html
You can get "relationship_matrics.csv" (Alfred objects relationship) & "A.csv" (Adjacency matrix)
```
cd graph_analysis/visual_genome
python visual_genome_scene_graphs_analysis.py 
```

## Create node & edge
You can get node & edge data at "./data_dgl"
```
cd graph_analysis
python create_dgl_data.py 
--object 
--verb
```

## analize verb in ALFRED
Prerequire
```
seaborn
scipy
```



---
---

# ALFRED

[<b>A Benchmark for Interpreting Grounded Instructions for Everyday Tasks</b>](https://arxiv.org/abs/1912.01734)  
[Mohit Shridhar](https://mohitshridhar.com/), [Jesse Thomason](https://jessethomason.com/), [Daniel Gordon](https://homes.cs.washington.edu/~xkcd/), [Yonatan Bisk](https://yonatanbisk.com/),  
[Winson Han](https://allenai.org/team.html), [Roozbeh Mottaghi](http://roozbehm.info/), [Luke Zettlemoyer](https://www.cs.washington.edu/people/faculty/lsz), [Dieter Fox](https://homes.cs.washington.edu/~fox/)  
[CVPR 2020](http://cvpr2020.thecvf.com/)

**ALFRED** (**A**ction **L**earning **F**rom **R**ealistic **E**nvironments and **D**irectives), is a new benchmark for learning a mapping from natural language instructions and egocentric vision to sequences of actions for household tasks. Long composition rollouts with non-reversible state changes are among the phenomena we include to shrink the gap between research benchmarks and real-world applications.

For the latest updates, see: [**askforalfred.com**](https://askforalfred.com)

## Citation

If you find the dataset or code useful, please cite:

```
@inproceedings{ALFRED20,
  title ={{ALFRED: A Benchmark for Interpreting Grounded
           Instructions for Everyday Tasks}},
  author={Mohit Shridhar and Jesse Thomason and Daniel Gordon and Yonatan Bisk and
          Winson Han and Roozbeh Mottaghi and Luke Zettlemoyer and Dieter Fox},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020},
  url  = {https://arxiv.org/abs/1912.01734}
}
```
