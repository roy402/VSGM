import os
import cv2
import torch
import numpy as np
import nn.vnn2 as VNN
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq_im_moca_semantic import Module as seq2seq_im_moca_semantic


class Module(seq2seq_im_moca_semantic):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab, importent_nodes=True)
        IMPORTENT_NDOES_FEATURE = self.config['semantic_cfg'].SCENE_GRAPH.EMBED_FEATURE_SIZE
        args_scene_graph = self.config['semantic_cfg'].SCENE_GRAPH
        if self.config['semantic_cfg'].SCENE_GRAPH.MODEL == "hete_gan":
            decoder = VNN.GANDec
        elif "DECODER" in self.config['semantic_cfg'].GENERAL and self.config['semantic_cfg'].GENERAL.DECODER == "softmax_gcn_Dec":
            decoder = VNN.softmax_gcn_Dec
        elif self.config['semantic_cfg'].GENERAL.DECODER == "PRIORI":
            decoder = VNN.PrioriDec
        elif self.config['semantic_cfg'].GENERAL.DECODER == "FeatWithoutFrame":
            decoder = VNN.FeatWithoutFrame
        elif self.config['semantic_cfg'].GENERAL.DECODER == "FeatWithoutFrameV2":
            decoder = VNN.FeatWithoutFrameV2
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V2":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH_V2
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V3":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH_V3
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V4":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH_V4
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V5":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH_V5
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V6":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH_V6
        elif self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V7":
            import nn.vnn5 as vnn
            decoder = vnn.Mini_MOCA_GRAPH_V7
        else:
            raise NotImplementedError()
        # else:
        #     decoder = vnn.ImportentNodes
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                           self.semantic_graph_implement, IMPORTENT_NDOES_FEATURE,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing)

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features (goal)
        if self.r_state['cont_lang_goal'] is None and self.r_state['enc_lang_goal'] is None:
            self.r_state['cont_lang_goal'], self.r_state['enc_lang_goal'] = self.encode_lang(feat)

        # encode language features (instr)
        if self.r_state['cont_lang_instr'] is None and self.r_state['enc_lang_instr'] is None:
            self.r_state['cont_lang_instr'], self.r_state['enc_lang_instr'] = self.encode_lang_instr(feat)

        # initialize embedding and hidden states (goal)
        if self.r_state['state_t_goal'] is None:
            self.r_state['state_t_goal'] = self.r_state['cont_lang_goal'], torch.zeros_like(self.r_state['cont_lang_goal'])

        # initialize embedding and hidden states (instr)
        if self.r_state['e_t'] is None and self.r_state['state_t_instr'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang_instr'].size(0), 1)
            self.r_state['state_t_instr'] = self.r_state['cont_lang_instr'], torch.zeros_like(self.r_state['cont_lang_instr'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']

        '''
        semantic graph
        '''

        if (self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V6" or self.config['semantic_cfg'].GENERAL.DECODER == "Mini_MOCA_GRAPH_V7")\
           and self.r_state['weighted_lang_t_goal'] is not None:
            state_t_goal, state_t_instr = self.r_state['weighted_lang_t_goal'], self.r_state['weighted_lang_t_instr']
            print("Mini_MOCA_GRAPH_V6")
        else:
            state_t_goal, state_t_instr = self.r_state['state_t_goal'], self.r_state['state_t_instr']
        # batch = 1
        all_meta_datas = feat['all_meta_datas']
        feat_global_graph = []
        feat_current_state_graph = []
        feat_history_changed_nodes_graph = []
        feat_priori_graph = []
        for env_index in range(len(all_meta_datas)):
            b_store_state = all_meta_datas[env_index]
            # get_meta_datas(cls, env, resnet):
            t_store_state = b_store_state["sgg_meta_data"]
            # cls.resnet.featurize([curr_image], batch=1).unsqueeze(0)
            t_store_state["rgb_image"] = feat['frames'][env_index, 0]
            global_graph_importent_features, current_state_graph_importent_features, history_changed_nodes_graph_importent_features, priori_importent_features,\
                global_graph_dict_objectIds_to_score, current_state_dict_objectIds_to_score, history_changed_dict_objectIds_to_score, priori_dict_dict_objectIds_to_score =\
                self.dec.store_and_get_graph_feature(t_store_state, env_index, state_t_goal, state_t_instr)
            feat_global_graph.append(global_graph_importent_features)
            feat_current_state_graph.append(current_state_graph_importent_features)
            feat_history_changed_nodes_graph.append(history_changed_nodes_graph_importent_features)
            feat_priori_graph.append(priori_importent_features)
        feat_global_graph = torch.cat(feat_global_graph, dim=0)
        feat_current_state_graph = torch.cat(feat_current_state_graph, dim=0)
        feat_history_changed_nodes_graph = torch.cat(feat_history_changed_nodes_graph, dim=0)
        feat_priori_graph = torch.cat(feat_priori_graph, dim=0)

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t_goal, state_t_instr, \
        lang_attn_t_goal, lang_attn_t_instr, subgoal_t, progress_t = \
            self.dec.step(
                self.r_state['enc_lang_goal'],
                self.r_state['enc_lang_instr'],
                feat['frames'][:, 0],
                e_t,
                self.r_state['state_t_goal'],
                self.r_state['state_t_instr'],
                feat_global_graph,
                feat_current_state_graph,
                feat_history_changed_nodes_graph,
                feat_priori_graph
            )

        # save states
        self.r_state['state_t_goal'] = state_t_goal
        self.r_state['state_t_instr'] = state_t_instr
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])
        self.r_state['weighted_lang_t_goal'] = lang_attn_t_goal
        self.r_state['weighted_lang_t_instr'] = lang_attn_t_instr

        assert len(all_meta_datas) == 1, "if not the analyze_graph object ind is error"
        global_graph_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            global_graph_dict_objectIds_to_score, graph_type="GLOBAL_GRAPH")
        current_state_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            current_state_dict_objectIds_to_score, graph_type="CURRENT_STATE_GRAPH")
        history_changed_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            history_changed_dict_objectIds_to_score, graph_type="HISTORY_CHANGED_NODES_GRAPH")
        priori_dict_ANALYZE_GRAPH = self.semantic_graph_implement.scene_graphs[0].analyze_graph(
            priori_dict_dict_objectIds_to_score, graph_type="PRIORI_GRAPH")

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)
        feat['out_subgoal_t'] = np.round(subgoal_t.view(-1).item(), decimals=2)
        feat['out_progress_t'] = np.round(progress_t.view(-1).item(), decimals=2)
        feat['global_graph_dict_ANALYZE_GRAPH'] = global_graph_dict_ANALYZE_GRAPH
        feat['current_state_dict_ANALYZE_GRAPH'] = current_state_dict_ANALYZE_GRAPH
        feat['history_changed_dict_ANALYZE_GRAPH'] = history_changed_dict_ANALYZE_GRAPH
        feat['priori_dict_ANALYZE_GRAPH'] = priori_dict_ANALYZE_GRAPH
        return feat
