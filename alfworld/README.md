# ALFWorld

## Command
textworld need to be download handly
```
pip install -r requirements.txt
apt install cmake g++ git make
pip install downward-faster_replan.zip
pip install TextWorld-handcoded_expert_integration.zip

export ALFWORLD_ROOT=/home/alfworld
export GRAPH_RCNN_ROOT=/home/graph-rcnn.pytorch/
python scripts/play_alfred_tw.py data/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ --domain data/alfred.pddl
```

### Train language
```
python dagger/train_dagger.py config/base_config.yaml
CUDA_VISIBLE_DEVICES=1 python dagger/train_butler_semantic_dagger.py config/base_config.yaml --semantic_config_file config/butler_memory_semantic_graph.yaml
```

### Train vision
```
python -m visdom.server
python dagger/train_vision_dagger.py config/vision_config.yaml
```

### INFOS
```
infos: {'admissible_commands': [['go to countertop 1', 'go to coffeemachine 1', 'go to cabinet 1', 'go to cabinet 2', 'go to cabinet 3', 'go to sink 1', 'go to cabinet 4', 'go to drawer 1', 'go to drawer 2', 'go to drawer 3', 'go to sinkbasin 1', 'go to cabinet 5', 'go to toaster 1', 'go to fridge 1', 'go to cabinet 6', 'go to cabinet 7', 'go to cabinet 8', 'go to microwave 1', 'go to cabinet 9', 'go to cabinet 10', 'go to cabinet 11', 'go to drawer 4', 'go to cabinet 12', 'go to stoveburner 1', 'go to drawer 5', 'go to stoveburner 2', 'inventory', 'look']], 'won': [False], 'goal_condition_success_rate': [0.0], 'extra.gamefile': ['../data/json_2.1.1/train/pick_and_place_simple-Lettuce-None-CounterTop-25/trial_T20190907_000040_764797'], 'expert_plan': [['go to fridge 1']]}
```

# Visual Semantic
$ALFWORLD_ROOT/agents/sgg/*
$ALFWORLD_ROOT/agents/semantic_graph/*
$ALFWORLD_ROOT/config setting
```
vision_dagger:
  model_type: "sgg"
```
object_classes = "__background__" + objects
predicate_to_ind = "__background__" + relations

## Visual Semantic Train Data
### Get train_sgg_vision_dagger_without_env training data
1. Get training data. use_exploration_frame_feats=False
"task_desc_string", "expert_action", "sgg_meta_data", "rgb_image",
```
python dagger/save_expert_data.py config/save_semantic_data_base.yaml --semantic_config_file config/save_semantic_data.yaml
```
2. Get exploration training data. use_exploration_frame_feats=True
"exploration_img", "exploration_sgg_meta_data"
```
python dagger/save_expert_data.py config/save_semantic_data_base.yaml --semantic_config_file config/save_semantic_data.yaml
```

### Hete semantic graph
#### train_sgg_vision_dagger_without_env
```
CUDA_VISIBLE_DEVICES=1 python dagger/train_sgg_vision_dagger_without_env.py config/hete_graph_base.yaml --semantic_config_file config/hete_semantic_graph.yaml
```
#### train_sgg_vision_dagger
update_per_k_game_steps: 5
replay_batch_size: 6
replay_sample_history_length: 4
```
CUDA_VISIBLE_DEVICES=1 python dagger/train_sgg_vision_dagger.py config/hete_graph_base.yaml --semantic_config_file config/hete_semantic_graph.yaml
```

### exploration frames and meta data first
```
CUDA_VISIBLE_DEVICES=1 python dagger/train_sgg_vision_dagger.py config/hete_graph_base.yaml --semantic_config_file config/exploration_hete_semantic_graph.yaml
```

### memory: Global Graph state, current info state, important node (choose node), all changed history nodes
```
CUDA_VISIBLE_DEVICES=1 python dagger/train_sgg_vision_dagger_without_env.py config/without_env_base.yaml --semantic_config_file config/memory_semantic_graph.yaml
```

## Analyze Graph
```
CUDA_VISIBLE_DEVICES=1 python dagger/train_sgg_vision_dagger.py config/analyze_base.yaml --semantic_config_file config/analyze_semantic_graph.yaml
```



# Eval
```
python eval/run_semantic_eval.py config/eval_semantic_config.yaml --semantic_config_file config/eval_semantic_graph.yaml

python eval/run_eval.py config/eval_config.yaml
```









## Citations

**ALFWorld**
```
@inproceedings{ALFWorld20,
  title ={{ALFWorld: Aligning Text and Embodied
           Environments for Interactive Learning}},
  author={Mohit Shridhar and Xingdi Yuan and
          Marc-Alexandre C\^ot\'e and Yonatan Bisk and
          Adam Trischler and Matthew Hausknecht},
  booktitle = {arXiv},
  year = {2020},
  url = {https://arxiv.org/abs/2010.03768}
}
```  

**ALFRED**
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

**TextWorld**
```
@inproceedings{cote2018textworld,
  title={Textworld: A learning environment for text-based games},
  author={C{\^o}t{\'e}, Marc-Alexandre and K{\'a}d{\'a}r, {\'A}kos and Yuan, Xingdi and Kybartas, Ben and Barnes, Tavian and Fine, Emery and Moore, James and Hausknecht, Matthew and El Asri, Layla and Adada, Mahmoud and others},
  booktitle={Workshop on Computer Games},
  pages={41--75},
  year={2018},
  organization={Springer}
}
```
