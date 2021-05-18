import os
import cv2
import torch
import numpy as np
import nn.vnn5 as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq_im_moca_semantic import Module as seq2seq_im_moca_semantic
import gen.constants as constants
# # 1 background + 108 object + 10
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


class Module(seq2seq_im_moca_semantic):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab, importent_nodes=True)
        IMPORTENT_NDOES_FEATURE = self.config['semantic_cfg'].SCENE_GRAPH.EMBED_FEATURE_SIZE
        if self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V1":
            decoder = vnn.MOCAMaskDepthGraph_V1
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V2":
            decoder = vnn.MOCAMaskDepthGraph_V2
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V3":
            decoder = vnn.MOCAMaskDepthGraph_V3
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V4":
            decoder = vnn.MOCAMaskDepthGraph_V4
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V5":
            decoder = vnn.MOCAMaskDepthGraph_V5
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskGraph_V1":
            decoder = vnn.MOCAMaskGraph_V1
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskGraph_V2":
            decoder = vnn.MOCAMaskGraph_V2
        else:
            print("self.config['semantic_cfg'].GENERAL.DECODER not found\n", self.config['semantic_cfg'].GENERAL.DECODER)
            return
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
        if "FEAT_NAME" in self.config['semantic_cfg'].GENERAL and self.config['semantic_cfg'].GENERAL.FEAT_NAME != "feat_conv.pt":
            self.feat_pt = self.config['semantic_cfg'].GENERAL.FEAT_NAME
        else:
            self.feat_pt = 'feat_depth_instance.pt'

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

        if self.r_state['weighted_lang_t_goal'] is None:
            self.r_state['weighted_lang_t_goal'] = self.r_state['cont_lang_goal'], torch.zeros_like(self.r_state['cont_lang_goal'])
            self.r_state['weighted_lang_t_instr'] = self.r_state['cont_lang_instr'], torch.zeros_like(self.r_state['cont_lang_instr'])

        '''
        semantic graph
        '''
        # batch = 1
        all_meta_datas = feat['all_meta_datas']
        feat_global_graph = []
        feat_current_state_graph = []
        feat_history_changed_nodes_graph = []
        feat_priori_graph = []
        for env_index in range(len(all_meta_datas)):
            b_store_state = all_meta_datas[env_index]
            global_graph_importent_features, current_state_graph_importent_features, history_changed_nodes_graph_importent_features, priori_importent_features,\
                global_graph_dict_objectIds_to_score, current_state_dict_objectIds_to_score, history_changed_dict_objectIds_to_score, priori_dict_dict_objectIds_to_score =\
                self.dec.store_and_get_graph_feature(b_store_state, feat, 0, env_index, self.r_state['weighted_lang_t_goal'], self.r_state['weighted_lang_t_instr'])
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
                {k: v[:, 0] for k, v in feat.items() if 'frames' in k}, # feat['frames'][:, 0],
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
        if self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V1":
            lang_attn_t_goal = lang_attn_t_goal
            lang_attn_t_instr = lang_attn_t_instr
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V2":
            lang_attn_t_goal = lang_attn_t_goal
            lang_attn_t_instr = lang_attn_t_instr
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V3":
            lang_attn_t_goal = state_t_goal
            lang_attn_t_instr = state_t_instr
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V4":
            lang_attn_t_goal = state_t_goal
            lang_attn_t_instr = state_t_instr
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V5":
            lang_attn_t_goal = state_t_goal
            lang_attn_t_instr = state_t_instr
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskGraph_V1":
            lang_attn_t_goal = state_t_goal
            lang_attn_t_instr = state_t_instr
        elif self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskGraph_V2":
            lang_attn_t_goal = lang_attn_t_goal
            lang_attn_t_instr = lang_attn_t_instr
        else:
            raise NotImplementedError()
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


    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

                # progress monitor supervision
                if self.args.pm_aux_loss_wt > 0:
                    num_actions = len([a for sg in ex['num']['action_low'] for a in sg])
                    subgoal_progress = [(i+1)/float(num_actions) for i in range(num_actions)]
                    feat['subgoal_progress'].append(subgoal_progress)

            #########
            # inputs
            #########

            # serialize segments
            self.serialize_lang_action(ex)

            # goal and instr language
            lang_goal, lang_instr = ex['num']['lang_goal'], ex['num']['lang_instr']

            # zero inputs if specified
            lang_goal = self.zero_input(lang_goal) if self.args.zero_goal else lang_goal
            lang_instr = self.zero_input(lang_instr) if self.args.zero_instr else lang_instr

            # append goal + instr
            feat['lang_goal'].append(lang_goal)
            feat['lang_instr'].append(lang_instr)

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])

                # low-level action mask
                if load_mask:
                    indices = []
                    for a in ex['plan']['low_actions']:
                        if a['api_action']['action'] in ['MoveAhead', 'LookUp', 'LookDown', 'RotateRight', 'RotateLeft']:
                            continue
                        if a['api_action']['action'] == 'PutObject':
                            label = a['api_action']['receptacleObjectId'].split('|')
                        else:
                            label = a['api_action']['objectId'].split('|')
                        indices.append(classes.index(label[4].split('_')[0] if len(label) >= 5 else label[0]))
                    feat['action_low_mask_label'].append(indices)

                # low-level valid interact
                feat['action_low_valid_interact'].append([a['valid_interact'] for a in ex['num']['action_low']])

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                all_meta_data = self._load_meta_data(root, ex["images"], device)
                feat['all_meta_datas'].append(all_meta_data)  # add stop frame

                im = torch.load(os.path.join(root, self.feat_pt))

                num_low_actions = len(ex['plan']['low_actions'])
                num_feat_frames = im["depth"].shape[0]

                if num_low_actions != num_feat_frames:
                    keep = [None] * len(ex['plan']['low_actions'])
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = i
                    keep.append(keep[-1])  # stop frame
                    feat['frames_depth_conv'].append(im["depth"][keep])
                    feat['frames_instance_conv'].append(im["instance"][keep])
                else:
                    feat['frames_depth_conv'].append(torch.cat([im["depth"], im["depth"][-1].unsqueeze(0)], dim=0))  # add stop frame
                    feat['frames_instance_conv'].append(torch.cat([im["instance"], im["instance"][-1].unsqueeze(0)], dim=0))  # add stop frame

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal', 'lang_instr'}:
                # language embedding and padding
                seqs = [torch.tensor(vv, device=device) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                embed_seq = self.emb_word(pad_seq)
                packed_input = pack_padded_sequence(embed_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'action_low_mask_label'}:
                # label
                seqs = torch.tensor([vvv for vv in v for vvv in vv], device=device, dtype=torch.long)
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            elif k in {'all_meta_datas'}:
                pass
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if ('frames' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        return feat

    def forward(self, feat, max_decode=300):
        cont_lang_goal, enc_lang_goal = self.encode_lang(feat)
        cont_lang_instr, enc_lang_instr = self.encode_lang_instr(feat)
        state_0_goal = cont_lang_goal, torch.zeros_like(cont_lang_goal)
        state_0_instr = cont_lang_instr, torch.zeros_like(cont_lang_instr)
        frames = {}
        for k, v in feat.items():
            if 'frames' in k:
                frames[k] = self.vis_dropout(feat[k])
        res = self.dec(enc_lang_goal, enc_lang_instr, frames, feat['all_meta_datas'], max_decode=max_decode, gold=feat['action_low'], state_0_goal=state_0_goal, state_0_instr=state_0_instr)
        feat.update(res)
        return feat


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        # if self.config['semantic_cfg'].GENERAL.DECODER == "MOCAMaskDepthGraph_V5":
        #     l_alow = feat['action_low'].view(-1)
        #     print("l_alow ", l_alow)
        #     print("out_action_low ", out['out_action_low'].view(-1, len(self.vocab['action_low'])))
        #     print("out_action_low_mask ", out['out_action_low_mask'])
        #     print("frames_depth_conv ", feat['frames_depth_conv'].shape)
        #     print("frames_instance_conv ", feat['frames_instance_conv'].shape)
        #     print("device ", feat['frames_instance_conv'].device)
        return super().compute_loss(out, batch, feat)