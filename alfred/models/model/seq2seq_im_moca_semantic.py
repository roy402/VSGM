import os
import cv2
import torch
import numpy as np
import nn.vnn2 as vnn
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model.seq2seq_moca_semantic import Module as Base
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
from PIL import Image
import gen.constants as constants
# # 1 background + 108 object + 10
classes = [0] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']
from nn.resnet import Resnet
'''
semantic
'''
import sys
import importlib
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT']))
from agents.utils import tensorboard
from agents.agent import oracle_sgg_dagger_agent
import json
import glob


class Module(Base):

    def __init__(self, args, vocab, importent_nodes=False):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)
        self.device = torch.device('cuda') if self.args.gpu else torch.device('cpu')
        '''
        semantic
        '''
        self.config = args.config_file
        self.config['general']['training']['batch_size'] = self.args.batch
        # for choose node attention input size
        self.config['general']['model']['block_hidden_dim'] = 2*args.dhid
        SEMANTIC_GRAPH_RESULT_FEATURE = self.config['semantic_cfg'].SCENE_GRAPH.RESULT_FEATURE
        # Semantic graph create
        self.semantic_graph_implement = oracle_sgg_dagger_agent.SemanticGraphImplement(self.config, self.args.gpu_id)

        # encoder and self-attention
        self.enc_goal = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_instr = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        self.enc_att_goal = vnn.SelfAttn(args.dhid*2)
        self.enc_att_instr = vnn.SelfAttn(args.dhid*2)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt > 0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        if not importent_nodes:
            decoder = vnn.ConvFrameMaskDecoderProgressMonitor
            self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid,
                               self.semantic_graph_implement, SEMANTIC_GRAPH_RESULT_FEATURE,
                               pframe=args.pframe,
                               attn_dropout=args.attn_dropout,
                               hstate_dropout=args.hstate_dropout,
                               actor_dropout=args.actor_dropout,
                               input_dropout=args.input_dropout,
                               teacher_forcing=args.dec_teacher_forcing)

        # dropouts
        self.vis_dropout = nn.Dropout(args.vis_dropout)
        self.lang_dropout = nn.Dropout(args.lang_dropout, inplace=True)
        self.input_dropout = nn.Dropout(args.input_dropout)

        # internal states
        self.state_t = None
        self.e_t = None
        self.test_mode = False

        # bce reconstruction loss
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'
        self.feat_exploration_pt = 'feat_exploration_conv.pt'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()
        self.extractor = None


    def finish_of_episode(self):
        self.semantic_graph_implement.reset_all_scene_graph()

    def _load_meta_data(self, root, list_img_traj, device):
        def sequences_to_one():
            print("_load with path", root)
            meta_datas = {
                "sgg_meta_data": [],
                "exploration_sgg_meta_data": [],
            }
            low_idx = -1
            for i, dict_frame in enumerate(list_img_traj):
                # 60 actions need 61 frames
                if low_idx != dict_frame["low_idx"]:
                    low_idx = dict_frame["low_idx"]
                else:
                    continue
                name_frame = dict_frame["image_name"].split(".")[0]
                file_path = os.path.join(root, "sgg_meta", name_frame + ".json")
                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        meta_data = json.load(f)
                    meta_data = {
                        "rgb_image": [],
                        "sgg_meta_data": meta_data,
                    }
                    meta_datas["sgg_meta_data"].append(meta_data)
                else:
                    print("file is not exist: {}".format(file_path))
            meta_datas["sgg_meta_data"].append(meta_data)
            exploration_path = os.path.join(root, "exploration_meta", "*.json")
            exploration_file_paths = glob.glob(exploration_path)
            for exploration_file_path in exploration_file_paths:
                with open(exploration_file_path, 'r') as f:
                    meta_data = json.load(f)
                meta_data = {
                    "exploration_sgg_meta_data": meta_data,
                }
                meta_datas["exploration_sgg_meta_data"].append(meta_data)
            return meta_datas
        all_meta_data_path = os.path.join(root, "all_meta_data.json")
        if os.path.isfile(all_meta_data_path):
            with open(all_meta_data_path, 'r') as f:
                all_meta_data = json.load(f)
        else:
            all_meta_data = sequences_to_one()
            with open(all_meta_data_path, 'w') as f:
                json.dump(all_meta_data, f)
        exporlation_ims = torch.load(os.path.join(root, self.feat_exploration_pt)).to(device)
        all_meta_data["exploration_imgs"] = exporlation_ims
        return all_meta_data

    def featurize(self, batch, load_mask=True, load_frames=True):
        '''
        tensorize and pad batch input
        '''
        device = self.device
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
                num_feat_frames = im.shape[0]

                if num_low_actions != num_feat_frames:
                    keep = [None] * len(ex['plan']['low_actions'])
                    for i, d in enumerate(ex['images']):
                        # only add frames linked with low-level actions (i.e. skip filler frames like smooth rotations and dish washing)
                        if keep[d['low_idx']] is None:
                            keep[d['low_idx']] = im[i]
                    keep.append(keep[-1])  # stop frame
                    feat['frames'].append(torch.stack(keep, dim=0))
                else:
                    feat['frames'].append(torch.cat([im, im[-1].unsqueeze(0)], dim=0))  # add stop frame

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


    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr'] for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low'] for a in a_group]


    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask


    def forward(self, feat, max_decode=300):
        cont_lang_goal, enc_lang_goal = self.encode_lang(feat)
        cont_lang_instr, enc_lang_instr = self.encode_lang_instr(feat)
        state_0_goal = cont_lang_goal, torch.zeros_like(cont_lang_goal)
        state_0_instr = cont_lang_instr, torch.zeros_like(cont_lang_instr)
        frames = self.vis_dropout(feat['frames'])
        res = self.dec(enc_lang_goal, enc_lang_instr, frames, feat['all_meta_datas'], max_decode=max_decode, gold=feat['action_low'], state_0_goal=state_0_goal, state_0_instr=state_0_instr)
        feat.update(res)
        return feat


    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang = feat['lang_goal']
        
        self.lang_dropout(emb_lang.data)
        
        enc_lang, _ = self.enc_goal(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        
        self.lang_dropout(enc_lang)
        
        cont_lang = self.enc_att_goal(enc_lang)

        return cont_lang, enc_lang

    def encode_lang_instr(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang = feat['lang_instr']
        
        self.lang_dropout(emb_lang.data)
        
        enc_lang, _ = self.enc_instr(emb_lang)
        enc_lang, _ = pad_packed_sequence(enc_lang, batch_first=True)
        
        self.lang_dropout(enc_lang)
        
        cont_lang = self.enc_att_instr(enc_lang)

        return cont_lang, enc_lang


    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t_goal': None,
            'state_t_instr': None,
            'e_t': None,
            'cont_lang_goal': None,
            'enc_lang_goal': None,
            'cont_lang_instr': None,
            'enc_lang_instr': None,
            'weighted_lang_t_goal': None,
            'weighted_lang_t_instr': None,
            'state_t_graph': None,
        }


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
        # batch = 1
        all_meta_datas = feat['all_meta_datas']
        feat_semantic_graph = []
        for env_index in range(len(all_meta_datas)):
            b_store_state = all_meta_datas[env_index]
            graph_embed_features, _, _ = \
                self.semantic_graph_implement.extract_visual_features(
                    store_state=b_store_state["sgg_meta_data"],
                    hidden_state=self.r_state['state_t_instr'][0][env_index:env_index+1],
                    env_index=env_index
                )
            # graph_embed_features is list (actually dont need list)
            feat_semantic_graph.append(graph_embed_features[0])
        feat_semantic_graph = torch.cat(feat_semantic_graph, dim=0)

        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t_goal, state_t_instr, \
        lang_attn_t_goal, lang_attn_t_instr, *_ = \
            self.dec.step(
                self.r_state['enc_lang_goal'],
                self.r_state['enc_lang_instr'],
                feat['frames'][:, 0],
                feat_semantic_graph=feat_semantic_graph,
                e_t=e_t,
                state_tm1_goal=self.r_state['state_t_goal'],
                state_tm1_instr=self.r_state['state_t_instr'],
            )

        # save states
        self.r_state['state_t_goal'] = state_t_goal
        self.r_state['state_t_instr'] = state_t_instr
        self.r_state['e_t'] = self.dec.emb(out_action_low.max(1)[1])

        # output formatting
        feat['out_action_low'] = out_action_low.unsqueeze(0)
        feat['out_action_low_mask'] = out_action_low_mask.unsqueeze(0)

        return feat


    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        '''
        output processing
        '''
        pred = {}
        for ex, alow, alow_mask in zip(batch, feat['out_action_low'].max(2)[1].tolist(), feat['out_action_low_mask']):
            # remove padding tokens
            if self.pad in alow:
                pad_start_idx = alow.index(self.pad)
                alow = alow[:pad_start_idx]
                alow_mask = alow_mask[:pad_start_idx]

            if clean_special_tokens:
                # remove <<stop>> tokens
                if self.stop_token in alow:
                    stop_start_idx = alow.index(self.stop_token)
                    alow = alow[:stop_start_idx]
                    alow_mask = alow_mask[:stop_start_idx]

            # index to API actions
            words = self.vocab['action_low'].index2word(alow)

            p_mask = [alow_mask[t].detach().cpu().numpy() for t in range(alow_mask.shape[0])]

            pred[self.get_task_and_ann_id(ex)] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
                'action_low_mask_label': p_mask,
                'action_navi_low': ".",
                'action_operation_low': ".",
                'action_navi_or_operation': [],
            }

        return pred


    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = self.device
        action_num = torch.tensor(self.vocab['action_low'].word2index(action), device=device)
        action_emb = self.dec.emb(action_num).unsqueeze(0)
        return action_emb


    def compute_loss(self, out, batch, feat):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # GT and predictions
        p_alow = out['out_action_low'].view(-1, len(self.vocab['action_low']))
        l_alow = feat['action_low'].view(-1)
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        p_alow = p_alow[pad_valid]
        l_alow = l_alow[pad_valid]
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt
        self.accuracy_metric(name="action_low", label=l_alow, predict=p_alow)

        # mask loss
        # valid_idxs = valid.view(-1).nonzero().view(-1)
        # flat_p_alow_mask = p_alow_mask.view(p_alow_mask.shape[0] * p_alow_mask.shape[1], p_alow_mask.shape[2])[valid_idxs]
        # losses['action_low_mask'] = self.ce_loss(flat_p_alow_mask, feat['action_low_mask_label']) * self.args.mask_loss_wt
        p_alow_mask = out['out_action_low_mask'].view(-1, len(classes))
        l_alow_mask_label = feat['action_low_mask_label'].view(-1)
        valid = feat['action_low_valid_interact'].view(-1)
        # mask label loss
        pad_valid = (valid != self.pad)
        p_alow_mask = p_alow_mask[pad_valid]
        losses['action_low_mask_label'] = self.ce_loss(p_alow_mask, l_alow_mask_label) * self.args.mask_loss_wt
        self.accuracy_metric(name="action_low_mask", label=l_alow_mask_label, predict=p_alow_mask)

        # subgoal completion loss
        if self.args.subgoal_aux_loss_wt > 0:
            p_subgoal = feat['out_subgoal'].squeeze(2)
            l_subgoal = feat['subgoals_completed']
            sg_loss = self.mse_loss(p_subgoal, l_subgoal)
            sg_loss = sg_loss.view(-1) * pad_valid.float()
            subgoal_loss = sg_loss.mean()
            losses['subgoal_aux'] = self.args.subgoal_aux_loss_wt * subgoal_loss

        # progress monitoring loss
        if self.args.pm_aux_loss_wt > 0:
            p_progress = feat['out_progress'].squeeze(2)
            l_progress = feat['subgoal_progress']
            pg_loss = self.mse_loss(p_progress, l_progress)
            pg_loss = pg_loss.view(-1) * pad_valid.float()
            progress_loss = pg_loss.mean()
            losses['progress_aux'] = self.args.pm_aux_loss_wt * progress_loss

        return losses


    def weighted_mask_loss(self, pred_masks, gt_masks):
        '''
        mask loss that accounts for weight-imbalance between 0 and 1 pixels
        '''
        bce = self.bce_with_logits(pred_masks, gt_masks)
        flipped_mask = self.flip_tensor(gt_masks)
        inside = (bce * gt_masks).sum() / (gt_masks).sum()
        outside = (bce * flipped_mask).sum() / (flipped_mask).sum()
        return inside + outside


    def flip_tensor(self, tensor, on_zero=1, on_non_zero=0):
        '''
        flip 0 and 1 values in tensor
        '''
        res = tensor.clone()
        res[tensor == 0] = on_zero
        res[tensor != 0] = on_non_zero
        return res


    def compute_metric(self, preds, data):
        '''
        compute f1 and extract match scores for output
        '''
        m = collections.defaultdict(list)
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            label = ' '.join([a['discrete_action']['action'] for a in ex['plan']['low_actions']])
            m['action_low_f1'].append(compute_f1(label.lower(), preds[i]['action_low'].lower()))
            m['action_low_em'].append(compute_exact(label.lower(), preds[i]['action_low'].lower()))
        return {k: sum(v)/len(v) for k, v in m.items()}