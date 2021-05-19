# VSGM - Enhance robot task understanding ability through visual semantic graph

## Prerequire
1. Please check `requirements.txt` to install python package.
2. (Optional) Build AI2-Thor simulation environment for headless
```
https://github.com/allenai/ai2thor
https://github.com/allenai/ai2thor-docker
```
3. (Optional) Download dataset following by alfred & Copy dataset to `VSGM/alfred/data/`
```
https://github.com/askforalfred/alfred
```
4. Set bash variables
```
cd VSGM/alfred
export ALFRED_ROOT=/home/VSGM/alfred/
export ALFWORLD_ROOT=/home/VSGM/alfworld/
export GRAPH_ANALYSIS=/home/VSGM/graph_analysis/
export GRAPH_RCNN_ROOT=/home/VSGM/agents/sgg/graph-rcnn.pytorch

# windows
SET ALFRED_ROOT=D:\alfred\alfred
SET ALFWORLD_ROOT=D:\alfred\alfworld
SET GRAPH_ANALYSIS=D:\alfred\graph_analysis
SET GRAPH_RCNN_ROOT=D:\alfred\alfworld\agents\sgg\graph-rcnn.pytorch
```
4. Download `Scene Graph Generation` Model
```
https://drive.google.com/file/d/1drMYv0dKYJpXMGo62s60eEcr6SL_N3F5/view?usp=sharing
```
5. Download Training dataset. Please check `README_Training_Data.md`.

## Training code
VSGM
```
python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/graph_map.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_graph_map --dout exp/graph_map --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1 --dframe 243 --sgg_pool 2 --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml --gpu_id cuda:1
```
Semantic Graph •* + SLAM
```
python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/graph_map_slam.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_graph_map --dout exp/graph_map_slam --splits data/splits/oct21.json --batch 2 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1 --dframe 243 --sgg_pool 2 --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml --gpu_id cuda:1
```
Semantic Graph •*
Please create train data from `README_Training_Data.md` `2. extract exploration img to resnet feature to feat_third_party_img_and_exploration.pt'`
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_mask_depth_graph_sgg_feat_v5.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_mini_mask_depth --dout exp/sgg_feat_mini_moca_test_mask_depth_graph_v5 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1
```
Oracle Semantic Graph •
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_mask_depth_graph_v5.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_mini_mask_depth --dout exp/mini_moca_test_mask_depth_graph_v5 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1
```
Oracle Semantic Graph
```
CUDA_VISIBLE_DEVICES=0 python models/train/train_semantic.py models/config/fast_epoch_base.yaml --semantic_config_file models/config/priori_semantic_graph.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_importent_nodes --dout exp/fast_moca --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --gpu --not_save_config
```
MOCA •
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_semantic.py models/config/without_env_base.yaml --semantic_config_file models/config/mini_moca_test_mask_depth_v1.yaml --data data/full_2.1.0/ --model seq2seq_im_moca_mini_test --dout exp/mini_test_mask_depth_v1 --splits data/splits/oct21.json --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --demb 100 --dhid 256 --not_save_config --gpu --task_types 1
```
Seq2Seq+PM
```
CUDA_VISIBLE_DEVICES=1 python models/train/train_seq2seq.py --data data/full_2.1.0/ --model seq2seq_im_mask --dout exp/1_seq2seq_im_mask_pm_and_subgoals_01 --splits data/splits/oct21.json --gpu --batch 5 --pm_aux_loss_wt 0.1 --subgoal_aux_loss_wt 0.1 --task_types 1 --demb 100 --dhid 256
```

## Evaluation code
```
cd $ALFRED_ROOT
python3 scripts/startx.py
# Pre-download eval MOCA maskrcnn model
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
# VSGM Model
https://drive.google.com/file/d/14j90vvcyKqcWqwTcLs55QAN2y-L02ilX/view?usp=sharing
# Semantic Graph •* + SLAM Model
https://drive.google.com/file/d/18jTNQpGkT66GXtxKC5_7gwQZFNkPHPKM/view?usp=sharing
```

VSGM
```
python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/graph_map_02-04-2021_16-19-10/best_seen.pth --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml --model seq2seq_im_moca_graph_map --data data/full_2.1.0/ --eval_split valid_seen --gpu --gpu_id 1 --task_types 1 --subgoals all
```
Semantic Graph •* + SLAM
```

```
Semantic Graph •*
```
CUDA_VISIBLE_DEVICES=1 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/sgg_feat_mini_moca_test_mask_depth_graph_v5_22-02-2021_15-43-23/best_seen.pth --model seq2seq_im_moca_mini_mask_depth --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all --sgg_config_file $GRAPH_RCNN_ROOT/configs/attribute.yaml

```
Oracle Semantic Graph •
```
CUDA_VISIBLE_DEVICES=1 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/mini_moca_test_mask_depth_graph_v5_18-02-2021_11-16-38/best_seen.pth --model seq2seq_im_moca_mini_mask_depth --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
```
Oracle Semantic Graph
```
CUDA_VISIBLE_DEVICES=1 python models/eval_moca/eval_semantic.py models/config/without_env_base.yaml --model_path exp/fast_priori_semantic_graph_28-01-2021_14-33-44/best_seen.pth --model seq2seq_im_moca_importent_nodes --data data/full_2.1.0/ --eval_split valid_unseen --gpu  --task_types 1 --subgoals all
```
MOCA •
```
CUDA_VISIBLE_DEVICES=1 python models/eval_thirdparty/eval_semantic.py models/config/without_env_base.yaml --model_path exp/mini_test_mask_depth_v1_12-02-2021_16-50-53/best_seen.pth --model seq2seq_im_moca_mini_test --data data/full_2.1.0/ --eval_split valid_seen --gpu  --task_types 1 --subgoals all
```
Seq2Seq+PM
```
python models/eval/eval_seq2seq.py --model_path exp/1_seq2seq_im_mask_pm_and_subgoals_01_25-02-2021_11-59-30/best_seen.pth --model models.model.seq2seq_im_mask --data data/full_2.1.0/ --eval_split valid_seen --gpu --gpu_id 1 --task_types 1 --subgoals all
```

## Other

### Scene Graph Generation
Read $ALFWORLD_ROOT/agents/sgg/TRAIN_SGG.md
Read $GRAPH_RCNN_ROOT/README.md

### Semantic Graph & Pytorch-Geometric
Read $ALFWORLD_ROOT/agents/semantic_graph/README.md

### Spatial Semantic Map
Read $ALFWORLD_ROOT/agents/graph_map/README.md