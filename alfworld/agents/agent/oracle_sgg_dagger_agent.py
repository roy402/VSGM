import os
import sys
import operator
from queue import PriorityQueue
import copy
import numpy as np
import torch
import torch.nn.functional as F
import modules.memory as memory
from agent import TextDAggerAgent
from modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from modules.layers import NegativeLogLoss, masked_mean, compute_mask
import torchvision.transforms as T
from torchvision import models
from torchvision.ops import boxes as box_ops
import importlib

sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
from agents.utils import tensorboard
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents', 'semantic_graph'))
import pdb
from semantic_graph import SceneGraph
from graph_map.graph_map import GraphMap
from graph_map.slam_map import SlamMAP
from sgg import alfred_data_format, sgg
from icecream import ic


class SemanticGraphImplement(torch.nn.Module):
    def __init__(self, config, device="cuda"):
        super(SemanticGraphImplement, self).__init__()
        '''
        NEW
        '''
        self.use_gpu = config['general']['use_cuda']
        self.use_exploration_frame_feats = config['vision_dagger']['use_exploration_frame_feats']

        # Semantic graph create
        self.cfg_semantic = config['semantic_cfg']
        self.ANALYZE_GRAPH = self.cfg_semantic.GENERAL.ANALYZE_GRAPH
        self.PRINT_DEBUG = self.cfg_semantic.GENERAL.PRINT_DEBUG
        self.isORACLE = self.cfg_semantic.SCENE_GRAPH.ORACLE
        self.EMBED_CURRENT_STATE = self.cfg_semantic.SCENE_GRAPH.EMBED_CURRENT_STATE
        self.EMBED_HISTORY_CHANGED_NODES = self.cfg_semantic.SCENE_GRAPH.EMBED_HISTORY_CHANGED_NODES
        self.RESULT_FEATURE = self.cfg_semantic.SCENE_GRAPH.RESULT_FEATURE
        ic(self.cfg_semantic.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE)

        # model
        self.graph_embed_model = importlib.import_module(self.cfg_semantic.SCENE_GRAPH.MODEL)
        self.graph_embed_model = self.graph_embed_model.Net(
            self.cfg_semantic,
            config=config,
            PRINT_DEBUG=self.PRINT_DEBUG
            )
        if self.use_gpu:
            # self.graph_embed_model.cuda()
            self.graph_embed_model.to(device)
        self.trans_MetaData = alfred_data_format.TransMetaData(
            self.cfg_semantic)
        self.scene_graphs = []
        self.graph_maps = []
        for i in range(config['general']['training']['batch_size']):
            scene_graph = SceneGraph(
                self.cfg_semantic,
                self.trans_MetaData.SGG_result_ind_to_classes,
                self.cfg_semantic.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE,
                device=device,
                )
            self.scene_graphs.append(scene_graph)
            if "GRAPH_MAP" in self.cfg_semantic:
                if "SLAM_MAP" in self.cfg_semantic and self.cfg_semantic.SLAM_MAP.USE_SLAM_MAP:
                    graph_map = SlamMAP(
                        self.cfg_semantic,
                        device=device,
                        )
                else:
                    graph_map = GraphMap(
                        self.cfg_semantic,
                        scene_graph.priori_features,
                        self.cfg_semantic.SCENE_GRAPH.NODE_INPUT_RGB_FEATURE_SIZE,
                        device=device,
                        object_classes_index_to_name=scene_graph.object_classes_index_to_name,
                        )
                self.graph_maps.append(graph_map)
        # initialize model
        # if not self.isORACLE:
        if not self.isORACLE or ("FEAT_NAME" in self.cfg_semantic.GENERAL and self.cfg_semantic.GENERAL.FEAT_NAME == "feat_sgg_depth_instance_test.pt"):
            self.cfg_sgg = config['sgg_cfg']
            # gpu = self.cfg_semantic.SGG.GPU
            gpu = self.cfg_semantic.SGG.GPU if "SGG" in self.cfg_semantic else 0
            self.detector = sgg.load_pretrained_model(
                self.cfg_sgg,
                self.trans_MetaData.transforms,
                self.trans_MetaData.SGG_result_ind_to_classes,
                "cuda:%d" % gpu,
                )
            self.detector.eval()
            self.detector.to(device="cuda:%d" % gpu)
        self.use_gpu = config['general']['use_cuda']

    def reset_all_scene_graph(self):
        global_graphs = []
        for scene_graph in self.scene_graphs:
            if self.PRINT_DEBUG:
                num_nodes_global_graph, num_nodes_current_state_graph, num_nodes_history_changed_nodes_graph = 0, 0, 0
                if scene_graph.global_graph.x is not None:
                    num_nodes_global_graph = len(scene_graph.global_graph.x)
                if scene_graph.current_state_graph.x is not None:
                    num_nodes_current_state_graph = len(scene_graph.current_state_graph.x)
                if scene_graph.history_changed_nodes_graph.x is not None:
                    num_nodes_history_changed_nodes_graph = len(scene_graph.history_changed_nodes_graph.x)
                global_graphs.append(
                    (
                        num_nodes_global_graph,
                        num_nodes_current_state_graph,
                        num_nodes_history_changed_nodes_graph
                    )
                )
            scene_graph.init_graph_data()
        for graph_map in self.graph_maps:
            graph_map.reset_map()
        if self.PRINT_DEBUG:
            # [(26, 13, 0), (80, 10, 2), (72, 4, 2), (25, 7, 1), (30, 5, 3), (27, 5, 1), (59, 14, 4), (30, 9, 1), (31, 6, 3), (61, 6, 0), (32, 4, 2), (36, 5, 2), (44, 14, 1), (19, 2, 2), (73, 15, 1), (77, 8, 4), (56, 9, 2), (59, 10, 4), (72, 5, 1), (16, 3, 0), (40, 13, 2), (22, 4, 3), (61, 9, 1), (23, 11, 1), (29, 6, 2), (90, 5, 1), (26, 13, 1), (93, 4, 3), (23, 14, 3), (37, 4, 2), (27, 7, 1), (69, 2, 6), (54, 1, 0), (34, 12, 3), (71, 21, 4), (95, 17, 1), (19, 7, 1), (25, 12, 1), (53, 6, 3), (28, 6, 3), (18, 9, 3), (44, 4, 1), (36, 5, 4), (34, 9, 0), (31, 3, 1), (29, 7, 1), (33, 5, 1), (59, 3, 2), (27, 8, 2), (13, 5, 1), (29, 15, 1), (32, 5, 1), (37, 10, 3), (65, 8, 2), (18, 13, 0), (75, 11, 0), (28, 4, 2), (58, 3, 0), (40, 5, 1), (29, 2, 1), (60, 9, 2), (27, 7, 3), (31, 14, 2), (32, 3, 3)]
            print("WARNING For DEBUG. \nglobal_graph nodes numbers result: \n", global_graphs)

    def get_env_last_event_data(self, thor):
        env = thor.env
        rgb_image = env.last_event.frame[:, :, ::-1]
        mask_image = env.last_event.instance_segmentation_frame
        sgg_meta_data = env.last_event.metadata['objects']
        store_state = {
            "rgb_image": rgb_image,
            "mask_image": mask_image,
            "sgg_meta_data": sgg_meta_data,
        }
        return store_state

    # for alfred model/nn
    def store_data_to_graph(self, thor=None, store_state=None, env_index=None, reset_current_graph=True, agent_meta=None, horizontal_view_angle=0, sgg_results=None):
        if thor is not None:
            store_state = self.get_env_last_event_data(thor)
        if store_state is None and self.isORACLE:
            raise NotImplementedError()

        scene_graph = self.scene_graphs[env_index]
        if self.isORACLE:
            rgb_image = store_state["rgb_image"]
            sgg_meta_data = store_state["sgg_meta_data"]
            target = self.trans_MetaData.trans_object_meta_data_to_relation_and_attribute(sgg_meta_data, agent_meta=agent_meta, horizontal_view_angle=horizontal_view_angle)
            scene_graph.add_oracle_local_graph_to_global_graph(rgb_image, target, reset_current_graph=reset_current_graph)
        else:
            sgg_result = sgg_results[0]
            scene_graph.add_local_graph_to_global_graph(None, sgg_result, reset_current_graph=reset_current_graph)

    def update_map(self, env_index, depth_image, agent_meta, sgg_results):
        graph_map = self.graph_maps[env_index]
        target = {
            "bbox": sgg_results[0]["bbox"],
            "labels": sgg_results[0]["labels"],
        }
        graph_map.update_map(
            np.array(depth_image.cpu().view(300, 300, 3)).astype(float),
            agent_meta,
            target)

    def get_map_feature(self, env_index, hidden_state=None):
        if "SLAM_MAP" in self.cfg_semantic and self.cfg_semantic.SLAM_MAP.USE_SLAM_MAP:
            graph_map = self.graph_maps[env_index]
            map_feature = graph_map.map_feature
            return map_feature, {}
        else:
            graph_map = self.graph_maps[env_index]
            importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.chose_importent_node_graph_map(
                graph_map.map,
                hidden_state,
            )
            return importent_node_feature, dict_objectIds_to_score
        # else:
        #     raise NotImplementedError

    def get_priori_feature(self, env_index, hidden_state):
        raise NotImplementedError
        scene_graph = self.scene_graphs[env_index]
        priori_graph = scene_graph.get_priori_graph()
        importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.priori_feature(
            priori_graph,
            hidden_state,
        )

    # for alfred model/nn
    def get_graph_feature(self, chose_type, env_index):
        scene_graph = self.scene_graphs[env_index]
        if chose_type == "GLOBAL_GRAPH":
            # embed graph data
            global_graph = scene_graph.get_graph_data()
            graph_feature, dict_objectIds_to_score = self.graph_embed_model(
                global_graph,
            )
        elif chose_type == "CURRENT_STATE_GRAPH":
            current_state_graph = scene_graph.get_current_state_graph_data()
            graph_feature, dict_objectIds_to_score = self.graph_embed_model(
                current_state_graph,
            )
        elif chose_type == "HISTORY_CHANGED_NODES_GRAPH":
            history_changed_nodes_graph = scene_graph.get_history_changed_nodes_graph_data()
            graph_feature, dict_objectIds_to_score = self.graph_embed_model(
                history_changed_nodes_graph,
            )
        elif chose_type == "PRIORI_GRAPH":
            priori_graph = scene_graph.get_priori_graph()
            graph_feature, dict_objectIds_to_score = self.graph_embed_model(
                priori_graph,
            )
        else:
            raise NotImplementedError
        return graph_feature, dict_objectIds_to_score

    # for alfred model/nn
    def chose_importent_node_feature(self, chose_type, env_index, hidden_state=None):
        scene_graph = self.scene_graphs[env_index]
        if chose_type == "GLOBAL_GRAPH":
            # embed graph data
            global_graph = scene_graph.get_graph_data()
            importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.chose_importent_node(
                global_graph,
                hidden_state,
            )
        elif chose_type == "CURRENT_STATE_GRAPH":
            current_state_graph = scene_graph.get_current_state_graph_data()
            importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.chose_importent_node(
                current_state_graph,
                hidden_state,
            )
        elif chose_type == "HISTORY_CHANGED_NODES_GRAPH":
            history_changed_nodes_graph = scene_graph.get_history_changed_nodes_graph_data()
            importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.chose_importent_node(
                history_changed_nodes_graph,
                hidden_state,
            )
        elif chose_type == "PRIORI_GRAPH":
            priori_graph = scene_graph.get_priori_graph()
            importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.chose_importent_node(
                priori_graph,
                hidden_state,
            )
        elif chose_type == "GRAPH_MAP":
            graph_map = self.graph_maps[env_index]
            importent_node_feature, dict_objectIds_to_score = self.graph_embed_model.chose_importent_node_graph_map(
                graph_map.map,
                hidden_state,
            )
        else:
            raise NotImplementedError
        return importent_node_feature, dict_objectIds_to_score

    # visual features for state representation
    def extract_visual_features(self, thor=None, store_state=None, hidden_state=None, env_index=None):
        if thor is not None:
            store_state = self.get_env_last_event_data(thor)
        if store_state is None:
            raise NotImplementedError()

        graph_embed_features = []
        scene_graph = self.scene_graphs[env_index]
        rgb_image = store_state["rgb_image"]
        if self.isORACLE:
            sgg_meta_data = store_state["sgg_meta_data"]
            target = self.trans_MetaData.trans_object_meta_data_to_relation_and_attribute(sgg_meta_data)
            scene_graph.add_oracle_local_graph_to_global_graph(rgb_image, target)
        else:
            rgb_image = rgb_image.unsqueeze(0)
            results = self.detector(rgb_image)
            result = results[0]
            scene_graph.add_local_graph_to_global_graph(rgb_image, result)
        # embed graph data
        global_graph = scene_graph.get_graph_data()
        graph_embed_feature, dict_ANALYZE_GRAPH = self.graph_embed_model(
            global_graph,
            CHOSE_IMPORTENT_NODE=True,
            hidden_state=hidden_state
        )
        if self.EMBED_CURRENT_STATE:
            current_state_graph = scene_graph.get_current_state_graph_data()
            current_state_feature, _ = self.graph_embed_model(
                current_state_graph,
            )
            graph_embed_feature = torch.cat([graph_embed_feature, current_state_feature], dim=1)
        if self.EMBED_HISTORY_CHANGED_NODES:
            history_changed_nodes_graph = scene_graph.get_history_changed_nodes_graph_data()
            history_changed_nodes_feature, _ = self.graph_embed_model(
                history_changed_nodes_graph,
            )
            graph_embed_feature = torch.cat([graph_embed_feature, history_changed_nodes_feature], dim=1)
        graph_embed_features.append(graph_embed_feature)

        dict_objectIds_to_score = []
        if self.ANALYZE_GRAPH:
            dict_objectIds_to_score = scene_graph.analyze_graph(dict_ANALYZE_GRAPH)
        return graph_embed_features, store_state, dict_objectIds_to_score

    def update_exploration_data_to_global_graph(self, exploration_transition_cache, env_index, exploration_imgs=None, agent_meta=None, horizontal_view_angle=0):
        if not self.use_exploration_frame_feats:
            return
        if self.PRINT_DEBUG:
            print("=== update_exploration_data_to_global_graph ===")
        scene_graph = self.scene_graphs[env_index]
        for i in range(len(exploration_transition_cache)):
            if exploration_imgs is not None:
                # _load_meta_data, all_meta_data["exploration_imgs"] = exporlation_ims
                rgb_image = exploration_imgs[i]
            else:
                rgb_image = exploration_transition_cache[i]["exploration_img"]
            if self.isORACLE:
                sgg_meta_data = exploration_transition_cache[i]["exploration_sgg_meta_data"]
                target = self.trans_MetaData.trans_object_meta_data_to_relation_and_attribute(sgg_meta_data, agent_meta=agent_meta, horizontal_view_angle=horizontal_view_angle)
                scene_graph.add_oracle_local_graph_to_global_graph(rgb_image, target)
                # print(len(sgg_meta_data))
                # print(self.scene_graphs[env_index].global_graph.x.shape)
            else:
                rgb_image = rgb_image.unsqueeze(0)
                results = self.detector(rgb_image)
                result = results[0]
                scene_graph.add_local_graph_to_global_graph(rgb_image, result)
        # import pdb; pdb.set_trace()


class OracleSggDAggerAgent(TextDAggerAgent):
    '''
    Vision Agent trained with DAgger
    '''
    def __init__(self, config):
        super().__init__(config)

        assert self.action_space == "generation"

        self.use_gpu = config['general']['use_cuda']
        self.transform = T.Compose([T.ToTensor()])
        '''
        semantic graph
        '''
        self.cfg_semantic = config['semantic_cfg']
        self.ANALYZE_GRAPH = self.cfg_semantic.GENERAL.ANALYZE_GRAPH
        self.PRINT_DEBUG = self.cfg_semantic.GENERAL.PRINT_DEBUG
        self.semantic_graph_implement = SemanticGraphImplement(config)

        # choose vision model
        self.vision_model_type = config['vision_dagger']['model_type']
        self.use_exploration_frame_feats = config['vision_dagger']['use_exploration_frame_feats']
        self.sequence_aggregation_method = config['vision_dagger']['sequence_aggregation_method']

        self.load_pretrained = self.cfg_semantic.GENERAL.LOAD_PRETRAINED
        self.load_from_tag = self.cfg_semantic.GENERAL.LOAD_PRETRAINED_PATH

        self.summary_writer = tensorboard.TensorBoardX(config["general"]["save_path"])

    def reset_all_scene_graph(self):
        self.semantic_graph_implement.reset_all_scene_graph()

    # visual features for state representation
    def extract_visual_features(self, thor=None, store_state=None, hidden_state=None, env_index=None):
        graph_embed_features, store_state, dict_objectIds_to_score = \
            self.semantic_graph_implement.extract_visual_features(
                thor=thor,
                store_state=store_state,
                hidden_state=hidden_state,
                env_index=env_index)
        return graph_embed_features, store_state, dict_objectIds_to_score

    def update_exploration_data_to_global_graph(self, exploration_transition_cache, env_index):
        self.semantic_graph_implement.update_exploration_data_to_global_graph(
            exploration_transition_cache,
            env_index
        )

    def finish_of_episode(self, episode_no, batch_size, decay_lr=False):
        super().finish_of_episode(episode_no, batch_size, decay_lr=decay_lr)
        self.semantic_graph_implement.reset_all_scene_graph()

    # without recurrency
    def train_dagger(self):
        raise NotImplementedError()

    # with recurrency
    def train_dagger_recurrent(self):
        if len(self.dagger_memory) < self.dagger_replay_batch_size:
            return None
        sequence_of_transitions = self.dagger_memory.sample_sequence_to_end(self.dagger_replay_batch_size, self.dagger_replay_sample_history_length)
        if sequence_of_transitions is None:
            return None

        if self.action_space == "generation":
            self.reset_all_scene_graph()
            losses = []
            for replay_batch_index in range(len(sequence_of_transitions)):
                sequence_of_transition = sequence_of_transitions[replay_batch_index]
                store_states = [transition[0] for transition in sequence_of_transition]
                task_desc_strings = [[transition[1]] for transition in sequence_of_transition]
                expert_actions = [[transition[3]] for transition in sequence_of_transition]
                env_index = replay_batch_index % len(self.semantic_graph_implement.scene_graphs)
                if env_index == 0:
                    self.reset_all_scene_graph()
                # exploration_data
                if self.use_exploration_frame_feats:
                    self.update_exploration_data_to_global_graph(
                        sequence_of_transition[0].exploration_data,
                        env_index
                    )
                # train model
                loss = self.train_command_generation_recurrent_teacher_force(
                    store_states,
                    task_desc_strings,
                    expert_actions,
                    train_now=False,
                    env_index=env_index,
                )
                if loss is not None:
                    losses.append(loss)
            if len(losses):
                loss = torch.stack(losses).mean()
                loss = self.grad(loss)
            else:
                loss = torch.tensor(0.)
            if self.PRINT_DEBUG:
                print("loss: ", loss.item())
            self.summary_writer.training_loss(train_loss=loss, optimizer=self.optimizer)
        else:
            raise NotImplementedError()

    '''
    from train_sgg_vision_dagger_without_env.py
    self.scene_graphs[i] only be use self.scene_graphs[0]
    '''
    # loss
    def train_command_generation_recurrent_teacher_force(self, store_state, seq_task_desc_strings, seq_target_strings, contains_first_step=False, train_now=True, env_index=None):
        '''
        store_state: list dict (one batch size), store_state[0].keys() dict_keys(['rgb_image', 'mask_image', 'sgg_meta_data'])
        seq_task_desc_strings: list list (n batch size), [['[SEP] put a candle in cabinet.'], ['[SEP] put a candle in cabinet.'], ['[SEP] put a candle in cabinet.'], ['[SEP] put a candle in cabinet.'], ['[SEP] put a candle in cabinet.'], ['[SEP] put a candle in cabinet.']]
        seq_target_strings: list list (n batch size), [['go to toilet 1'], ['take candle 1 from toilet 1'], ['go to cabinet 3'], ['open cabinet 3'], ['put candle 1 in/on cabinet 3'], ['close cabinet 3']]
        '''
        loss_list = []
        previous_dynamics = None
        batch_size = len(seq_target_strings[0])
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
        # with torch.autograd.set_detect_anomaly(True):
        for step_no in range(len(seq_target_strings)):
            input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in seq_target_strings[step_no]]
            output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in seq_target_strings[step_no]]
            observation_feats, _, _ = self.extract_visual_features(
                store_state=store_state[step_no],
                hidden_state=previous_dynamics,
                env_index=env_index)

            obs = [o.to(h_td.device) for o in observation_feats]
            aggregated_obs_feat = self.aggregate_feats_seq(obs)
            # import pdb; pdb.set_trace()
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            averaged_vision_td_representation = self.online_net.masked_mean(
                vision_td, vision_td_mask)
            current_dynamics = self.online_net.rnncell(
                averaged_vision_td_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_vision_td_representation)

            input_target = self.get_word_input(input_target_strings)
            ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            # torch.Size([1, 5, 28996])
            pred = self.online_net.vision_decode(
                input_target, target_mask, vision_td, vision_td_mask, current_dynamics)  # batch x target_length x vocab
            # pred_ind = torch.argmax(pred, dim=2)
            # values, topk_indices = torch.topk(pred, 3, dim=2)
            # print("\nground_truth: {}\npred: {}\noutput_target_strings: {}".format(
            #     ground_truth.to('cpu').numpy(), values.detach().to('cpu').numpy(), output_target_strings))
            # print("\nground_truth: {}\noutput_target_strings: {}".format(
            #     ground_truth.to('cpu').numpy(), output_target_strings))
            # self.one_hot_embed_to_word(topk_indices)

            previous_dynamics = current_dynamics

            batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
            loss = torch.mean(batch_loss)
            loss_list.append(loss)

        if loss_list is None:
            return None
        loss = torch.stack(loss_list).mean()
        # print("loss: ", loss)
        if train_now:
            loss = self.grad(loss)
            return loss
        else:
            return loss

    def grad(self, loss):
        # Backpropagate
        self.online_net.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.clip_grad_norm)
        self.optimizer.step()  # apply gradients
        return to_np(loss)

    def one_hot_embed_to_word(self, topk_indices):
        batch_word_list = np.array(topk_indices.to('cpu'))
        for batch in batch_word_list:
            for b in batch.T:
                res = self.tokenizer.decode(b)
                res = res.replace("[CLS]", "").split('[SEP]')[0]
                res = res.replace(" in / on ", " in/on ")
                res = res.encode('utf-8')
                print("res: ", res)

    # recurrent
    def command_generation_greedy_generation(self, observation_feats, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            # print(observation_feats to word)
            batch_size = len(observation_feats)

            # torch.Size([1, 1024])
            aggregated_obs_feat = self.aggregate_feats_seq(observation_feats)
            # torch.Size([1, 1, 64])
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            # ['[SEP] clean some potato and put it in garbagecan.']
            # torch.Size([1, 14, 64])
            # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.]])
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
            h_obs = h_obs.to(h_td_mean.device)
            # torch.Size([1, 2, 64])
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            if self.recurrent:
                averaged_vision_td_representation = self.online_net.masked_mean(vision_td, vision_td_mask)
                current_dynamics = self.online_net.rnncell(averaged_vision_td_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_vision_td_representation)
            else:
                current_dynamics = None

            # greedy generation
            input_target_list = [[self.word2id["[CLS]"]] for i in range(batch_size)]
            eos = np.zeros(batch_size)
            for _ in range(self.max_target_length):
                input_target = copy.deepcopy(input_target_list)
                input_target = pad_sequences(input_target, maxlen=max_len(input_target)).astype('int32')
                input_target = to_pt(input_target, self.use_cuda)
                target_mask = compute_mask(input_target)  # mask of ground truth should be the same
                # tensor([[[4.1473e-05, 3.1972e-05, 3.8759e-05,  ..., 3.6035e-05, 3.4417e-05, 6.3384e-05]]])
                # torch.Size([1, 1, 28996])
                pred = self.online_net.vision_decode(input_target, target_mask, vision_td, vision_td_mask, current_dynamics)  # batch x target_length x vocab
                # pointer softmax
                pred = to_np(pred[:, -1])  # batch x vocab
                pred = np.argmax(pred, -1)  # batch
                for b in range(batch_size):
                    new_stuff = [pred[b]] if eos[b] == 0 else []
                    input_target_list[b] = input_target_list[b] + new_stuff
                    if pred[b] == self.word2id["[SEP]"]:
                        eos[b] = 1
                if np.sum(eos) == batch_size:
                    break
            res = [self.tokenizer.decode(item) for item in input_target_list]
            res = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in res]
            res = [item.replace(" in / on ", " in/on " ) for item in res]
            return res, current_dynamics

    def command_generation_beam_search_generation(self, observation_feats, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            batch_size = len(observation_feats)
            beam_width = self.beam_width
            if beam_width == 1:
                res, current_dynamics = self.command_generation_greedy_generation(observation_feats, task_desc_strings, previous_dynamics)
                res = [[item] for item in res]
                return res, current_dynamics
            generate_top_k = self.generate_top_k
            res = []

            aggregated_obs_feat = self.aggregate_feats_seq(observation_feats)
            h_obs = self.online_net.vision_fc(aggregated_obs_feat)
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            h_td_mean = self.online_net.masked_mean(h_td, td_mask).unsqueeze(1)
            h_obs = h_obs.to(h_td_mean.device)
            vision_td = torch.cat((h_obs, h_td_mean), dim=1) # batch x k boxes x hid
            vision_td_mask = torch.ones((batch_size, h_obs.shape[1]+h_td_mean.shape[1])).to(h_td_mean.device)

            averaged_vision_td_representation = self.online_net.masked_mean(vision_td, vision_td_mask)

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(averaged_vision_td_representation, vision_td_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):

                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = aggregated_obs_feat[b: b + 1]  # 1 x obs_len
                __obs_mask = td_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = averaged_vision_td_representation[b: b + 1]  # 1 x obs_len x hid
                if current_dynamics is not None:
                    __current_dynamics = current_dynamics[b: b + 1]  # 1 x hid
                else:
                    __current_dynamics = None
                ended_nodes = []

                # starting node -  previous node, input target, logp, length
                node = BeamSearchNode(None, __input_target_list, 0, 1)
                nodes_queue = PriorityQueue()
                # start the queue
                nodes_queue.put((node.val, node))
                queue_size = 1

                while(True):
                    # give up when decoding takes too long
                    if queue_size > 2000:
                        break

                    # fetch the best node
                    score, n = nodes_queue.get()
                    __input_target_list = n.input_target

                    if (n.input_target[-1] == self.word2id["[SEP]"] or n.length >= self.max_target_length) and n.previous_node != None:
                        ended_nodes.append((score, n))
                        # if we reached maximum # of sentences required
                        if len(ended_nodes) >= generate_top_k:
                            break
                        else:
                            continue

                    input_target = pad_sequences([__input_target_list], dtype='int32')
                    input_target = to_pt(input_target, self.use_cuda)
                    target_mask = compute_mask(input_target)
                    # decode for one step using decoder
                    pred = self.online_net.decode(input_target, target_mask, __aggregated_obs_representation, __obs_mask, __current_dynamics, __input_obs)  # 1 x target_length x vocab
                    pred = pred[0][-1].cpu()
                    gt_zero = torch.gt(pred, 0.0).float()  # vocab
                    epsilon = torch.le(pred, 0.0).float() * 1e-8  # vocab
                    log_pred = torch.log(pred + epsilon) * gt_zero  # vocab

                    top_beam_width_log_probs, top_beam_width_indicies = torch.topk(log_pred, beam_width)
                    next_nodes = []

                    for new_k in range(beam_width):
                        pos = top_beam_width_indicies[new_k]
                        log_p = top_beam_width_log_probs[new_k].item()
                        node = BeamSearchNode(n, __input_target_list + [pos], n.log_prob + log_p, n.length + 1)
                        next_nodes.append((node.val, node))

                    # put them into queue
                    for i in range(len(next_nodes)):
                        score, nn = next_nodes[i]
                        nodes_queue.put((score, nn))
                    # increase qsize
                    queue_size += len(next_nodes) - 1

                # choose n best paths
                if len(ended_nodes) == 0:
                    ended_nodes = [nodes_queue.get() for _ in range(generate_top_k)]

                utterances = []
                for score, n in sorted(ended_nodes, key=operator.itemgetter(0)):
                    utte = n.input_target
                    utte_string = self.tokenizer.decode(utte)
                    utterances.append(utte_string)
                utterances = [item.replace("[CLS]", "").replace("[SEP]", "").strip() for item in utterances]
                utterances = [item.replace(" in / on ", " in/on " ) for item in utterances]
                res.append(utterances)
            return res, current_dynamics

    def get_vision_feat_mask(self, observation_feats):
        batch_size = len(observation_feats)
        num_vision_feats = [of.shape[0] for of in observation_feats]
        max_feat_len = max(num_vision_feats)
        mask = torch.zeros((batch_size, max_feat_len))
        for b, num_vision_feat in enumerate(num_vision_feats):
            mask[b,:num_vision_feat] = 1
        return mask

    def extract_exploration_frame_feats(self, exploration_frames):
        exploration_frame_feats = []
        for batch in exploration_frames:
            ef_feats = []
            for image in batch:
                raise NotImplementedError()
                # observation_feats, _ = self.extract_visual_features(envs=env.envs, store_state=store_state[step_no])
                ef_feats.append(self.extract_visual_features([image])[0])
            # cat_feats = torch.cat(ef_feats, dim=0)
            max_feat_len = max([f.shape[0] for f in ef_feats])
            stacked_feats = self.online_net.vision_fc.pad_and_stack(ef_feats, max_feat_len=max_feat_len)
            stacked_feats = stacked_feats.view(-1, self.online_net.vision_fc.in_features)
            exploration_frame_feats.append(stacked_feats)
        return exploration_frame_feats

    def aggregate_feats_seq(self, feats):
        if self.sequence_aggregation_method == "sum":
            return [f.sum(0).unsqueeze(0) for f in feats]
        elif self.sequence_aggregation_method == "average":
            return [f.mean(0).unsqueeze(0) for f in feats]
        elif self.sequence_aggregation_method == "rnn":
            max_feat_len = max([f.shape[0] for f in feats])
            feats_stack = self.online_net.vision_fc.pad_and_stack(feats, max_feat_len=max_feat_len)
            feats_h, feats_c = self.online_net.vision_feat_seq_rnn(feats_stack)
            aggregated_feats = feats_h[:,0,:].unsqueeze(1)
            return [b for b in aggregated_feats]
        else:
            raise ValueError("sequence_aggregation_method must be sum, average or rnn")
