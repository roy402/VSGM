import os
from sys import platform
import torch
import numpy as np
import nn.vnn as vnn
import nn.resnet as resnet
import collections
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torchvision import transforms
from model.contrastive_pretrain import Module as Base
from model.gcn import GCN, GCNVisual
from model.dgl_gcn_hete import NetGCN, HETLOWSG
from models.utils.metric import compute_f1, compute_exact
from gen.utils.image_util import decompress_mask
import cv2
from PIL import Image
import pdb
trans_color = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1),
    transforms.ToTensor(),
])
trans_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
fn_cos = nn.CosineSimilarity(dim=0, eps=1e-6)
fn_logsoftmax = nn.LogSoftmax(dim=1)


class Module(Base):

    def __init__(self, args, vocab):
        '''
        Seq2Seq agent
        '''
        super().__init__(args, vocab)

        # encoder and self-attention
        # [batch, sentance, out_feat]
        self.device = torch.device(
            "cuda:%d" % args.gpu_id if torch.cuda.is_available() and args.gpu else "cpu")
        assert args.dframe == 1000, "resnet output 1000 predict"
        self.enc = nn.LSTM(args.demb, args.dhid, bidirectional=True, batch_first=True)
        # self.enc.to(torch.device("cpu"))
        self.visual_encoder = resnet.Resnet2(args, torch.device("cpu"), DataParallelDevice=args.DataParallelDevice)
        self.LSTM_visual_encoder = nn.LSTM(args.dframe + 300, args.dhid*2, batch_first=True)
        self.mlp_nlp_enc = nn.Linear(args.dhid*2, args.dhid*2)
        self.mlp_vis_enc = nn.Linear(args.dhid*2, args.dhid*2)
        self.enc_att = vnn.SelfAttn(args.dhid*2, DataParallelDevice=args.DataParallelDevice)
        self.enc_visual_att = vnn.SelfAttn(args.dhid*2, DataParallelDevice=args.DataParallelDevice)
        self._to_DataParallel(DataParallelDevice=args.DataParallelDevice)
        # self.enc_att = vnn.SelfAttn(args.dhid*2, is_DataParallel=False)
        # self.enc_visual_att = vnn.SelfAttn(args.dhid*2, is_DataParallel=False)

        # subgoal monitoring
        self.subgoal_monitoring = (self.args.pm_aux_loss_wt >
                                   0 or self.args.subgoal_aux_loss_wt > 0)

        # frame mask decoder
        # decoder = vnn.ConvFrameMaskDecoderProgressMonitor if self.subgoal_monitoring else vnn.ConvFrameMaskDecoder
        decoder = vnn.DataParallelDecoder if args.DataParallelDevice is not None else vnn.ConvFrameMaskAttentionDecoderProgressMonitor
        self.dec = decoder(self.emb_action_low, args.dframe, 2*args.dhid, args.dgcnout,
                           pframe=args.pframe,
                           attn_dropout=args.attn_dropout,
                           hstate_dropout=args.hstate_dropout,
                           actor_dropout=args.actor_dropout,
                           input_dropout=args.input_dropout,
                           teacher_forcing=args.dec_teacher_forcing,
                           visual_encode=False,
                           DataParallelDevice=args.DataParallelDevice)
        self.dec = self.dec.to(device=self.device)

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

        # paths
        self.root_path = os.getcwd()
        self.feat_pt = 'feat_conv.pt'
        self.feat_raw_pt = 'feat_raw_frames.pt'
        self.fold_frames = 'raw_images'
        self.feat_depth_pt = 'feat_depth.pt'
        self.fold_depth = 'depth_images'

        # params
        self.max_subgoals = 25

        # reset model
        self.reset()

    def _to_DataParallel(self, DataParallelDevice=None):
        if DataParallelDevice:
            # self.enc = torch.nn.DataParallel(self.enc, device_ids=DataParallelDevice)
            # self.LSTM_visual_encoder = torch.nn.DataParallel(self.LSTM_visual_encoder, device_ids=DataParallelDevice)
            pass

    def featurize(self, batch, load_mask=True, load_frames=True, augmentation=False):
        '''
        tensorize and pad batch input
        '''
        device = torch.device("cuda:%d" %
                              self.args.gpu_id) if self.args.gpu else torch.device('cpu')
        feat = collections.defaultdict(list)

        for ex in batch:
            ###########
            # auxillary
            ###########

            if not self.test_mode:
                # subgoal completion supervision
                if self.args.subgoal_aux_loss_wt > 0:
                    feat['subgoals_completed'].append(
                        np.array(ex['num']['low_to_high_idx']) / self.max_subgoals)

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
            lang_goal_instr = lang_goal + lang_instr
            # goal and instr language
            lang_goal, lang_instr = ex['ann']['goal'], ex['ann']['instr']

            # append goal + instr
            lang_goal_instr = []
            for string in lang_goal:
                lang_goal_instr.append(string.strip())
            for list_string in lang_instr:
                lang_goal_instr.extend([string.strip() for string in list_string])
            feat['lang_goal_instr'].append(lang_goal_instr)

            # load Resnet features from disk
            if load_frames and not self.test_mode:
                root = self.get_task_root(ex)
                path_img_frames = os.path.join(root, self.fold_frames)
                self._load_img(feat, path_img_frames, "frames",
                               ex["images"], self.feat_raw_pt, ".jpg", augmentation=augmentation)
                # import pdb; pdb.set_trace()
                # depth
                # path_img_depth = os.path.join(root, self.feat_depth)
                # self._load_img(feat, path_img_depth, "frames_depth", ex["images"])

            #########
            # outputs
            #########

            if not self.test_mode:
                # low-level action
                feat['action_low'].append([a['action'] for a in ex['num']['action_low']])
                # embedding
                actions = [a['api_action']['action'] for a in ex['plan']['low_actions']]
                actions.extend(["stop"])
                embedding_actions = self.get_faxtText_embedding(actions).clone().detach()
                feat['embedding_action_low'].append(embedding_actions)
                # low-level action mask
                if load_mask:
                    feat['action_low_mask'].append([self.decompress_mask(
                        a['mask']) for a in ex['num']['action_low'] if a['mask'] is not None])

                # low-level valid interact
                feat['action_low_valid_interact'].append(
                    [a['valid_interact'] for a in ex['num']['action_low']])

        # tensorization and padding
        for k, v in feat.items():
            if k in {'lang_goal_instr'}:
                # language embedding and padding
                seqs = [self.get_faxtText_embedding(vv).to(device=device).clone().detach() for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                seq_lengths = np.array(list(map(len, v)))
                packed_input = pack_padded_sequence(
                    pad_seq, seq_lengths, batch_first=True, enforce_sorted=False)
                feat[k] = packed_input
            elif k in {'action_low_mask'}:
                # mask padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                feat[k] = seqs
            elif k in {'subgoal_progress', 'subgoals_completed'}:
                # auxillary padding
                seqs = [torch.tensor(vv, device=device, dtype=torch.float) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq
            else:
                # default: tensorize and pad sequence
                seqs = [torch.tensor(vv, device=device, dtype=torch.float if (
                    'frames' in k or 'embedding_action_low' in k) else torch.long) for vv in v]
                pad_seq = pad_sequence(seqs, batch_first=True, padding_value=self.pad)
                feat[k] = pad_seq

        # import pdb; pdb.set_trace()
        # len(feat['action_low'][0])
        # feat['frames'][0].shape
        return feat

    def serialize_lang_action(self, feat):
        '''
        append segmented instr language and low-level actions into single sequences
        '''
        is_serialized = not isinstance(feat['num']['lang_instr'][0], list)
        if not is_serialized:
            feat['num']['lang_instr'] = [word for desc in feat['num']['lang_instr']
                                         for word in desc]
            if not self.test_mode:
                feat['num']['action_low'] = [a for a_group in feat['num']['action_low']
                                             for a in a_group]

    def decompress_mask(self, compressed_mask):
        '''
        decompress mask from json files
        '''
        # import pdb; pdb.set_trace()
        mask = np.array(decompress_mask(compressed_mask))
        mask = np.expand_dims(mask, axis=0)
        return mask

    def forward(self, feat, max_decode=300):
        cont_lang, enc_lang = self.encode_lang(feat)
        state_0 = cont_lang, torch.zeros_like(cont_lang)
        # import pdb; pdb.set_trace()
        """
        resnet feature
        """
        frames = self.vis_dropout(feat['frames'])
        list_feature_visal = [self.visual_encoder(frame, self.device) for frame in frames]
        feature_visal = torch.stack(list_feature_visal)
        gcn_embedding = self.gcn(frames)
        res = self.dec(enc_lang, feature_visal, gcn_embedding, max_decode=max_decode,
                       gold=feat['action_low'], state_0=state_0)
        feat.update(res)
        return feat

    def encode_lang(self, feat):
        '''
        encode goal+instr language
        '''
        emb_lang_goal_instr = feat['lang_goal_instr']
        self.lang_dropout(emb_lang_goal_instr.data)
        enc_lang_goal_instr, _ = self.enc(emb_lang_goal_instr)
        enc_lang_goal_instr, _ = pad_packed_sequence(enc_lang_goal_instr, batch_first=True)
        self.lang_dropout(enc_lang_goal_instr)
        cont_lang_goal_instr = self.enc_att(enc_lang_goal_instr)

        return cont_lang_goal_instr, enc_lang_goal_instr

    def _encode_visual(self, frames, embedding_actions):
        '''
        frames: torch.Size([3, 138, 3, 224, 224])
        '''
        list_feature_visal = [self.visual_encoder(frame, self.device) for frame in frames]
        # torch.Size([3, 138, 1000])
        feature_visal = torch.stack(list_feature_visal)
        # torch.Size([3, 138, 1300])
        feature = torch.cat([feature_visal, embedding_actions], dim=2)
        # torch.Size([3, 138, 1024])
        feature_visal, (hn, cn) = self.LSTM_visual_encoder(feature)
        # torch.Size([3, 1024])
        feature_visal = self.enc_visual_att(feature_visal)
        # import pdb; pdb.set_trace()
        return feature_visal

    def forward_visaul_action_instruction(self, feat):
        feature_ins, _ = self.encode_lang(feat)
        frames = self.vis_dropout(feat['frames'])
        embedding_actions = feat['embedding_action_low']
        feature_visal = self._encode_visual(frames, embedding_actions)
        feature_ins = self.mlp_nlp_enc(feature_ins)
        feature_visal = self.mlp_vis_enc(feature_visal)
        return feature_visal, feature_ins

    def compute_SimCLR_loss(self, feature):
        '''
        loss column 0 is pos similarity
        '''
        list_sim = []
        # 6
        for i in range(0, feature.shape[0], 3):
            current = i
            pos_samples = i + 1
            neg_samples = [*range(feature.shape[0])]
            del neg_samples[pos_samples], neg_samples[current]
            i_cos_compare = [fn_cos(feature[current], feature[pos_samples])]
            # [0, 1, 2, 5]
            if self.args.print:
                print(neg_samples)
            for neg_sample in neg_samples:
                i_cos_compare.append(fn_cos(feature[current], feature[neg_sample]))
            # tensor([0.9845, 0.8050, 0.8093, 0.9821, 0.9845], grad_fn=<StackBackward>)
            i_cos_compare = torch.stack(i_cos_compare)
            list_sim.append(i_cos_compare)
        # tensor([[0.9819, 0.8073, 0.8050, 0.8097, 0.7932],
        #         [0.9845, 0.8050, 0.8093, 0.9821, 0.9789]], grad_fn=<StackBackward>)
        # torch.Size([2, 5])
        all_sim = torch.stack(list_sim)
        # tensor([[1.4696, 1.6442, 1.6465, 1.6418, 1.6583],
        #         [1.5405, 1.7200, 1.7157, 1.5429, 1.5461]], grad_fn=<NegBackward>)
        result = -fn_logsoftmax(all_sim)
        # loss column 0 is pos similarity
        loss = result[:, 0].sum()
        return loss

    def instruction_instruction_contrastive(self, feature_ins):
        loss_pos, loss_neg = 0, 0
        for i in range(0, feature_ins.shape[0], 3):
            current = i
            positive_sample = i + 1
            negative_sample = i + 2
            loss_pos += fn_cos(feature_ins[current], feature_ins[positive_sample])
            loss_neg += -fn_cos(feature_ins[current], feature_ins[negative_sample])
        return loss_pos, loss_neg

    def visaul_instruction_contrastive(self, feature_visal, feature_ins):
        loss_pos, loss_neg = 0, 0
        for i in range(0, feature_visal.shape[0], 3):
            current = i
            positive_sample = i + 1
            negative_sample = i + 2
            loss_pos += fn_cos(feature_visal[current], feature_ins[current]) + \
                fn_cos(feature_visal[current], feature_ins[positive_sample])
            loss_neg += -fn_cos(feature_visal[current], feature_ins[negative_sample])

            # loss_sum += torch.dist(feature_visal[current], feature_ins[current]) + \
            #     torch.dist(feature_visal[current], feature_ins[positive_sample]) - \
            #     torch.dist(feature_visal[current], feature_ins[negative_sample])
        return loss_pos, loss_neg

    def visaul_visaul_contrastive(self, feature_visal, feature_ins):
        loss_pos, loss_neg = 0, 0
        for i in range(0, feature_visal.shape[0], 3):
            current = i
            positive_sample = i + 1
            negative_sample = i + 2
            loss_pos += fn_cos(feature_visal[current], feature_visal[positive_sample])
            loss_neg += -fn_cos(feature_visal[current], feature_visal[negative_sample])
            # loss_sum += torch.dist(feature_visal[current], feature_visal[positive_sample]) - \
            #     torch.dist(feature_visal[current], feature_visal[negative_sample])
        return loss_pos, loss_neg

    def _load_img(self, feat, root, key_name, list_img_traj, name_pt, type_image=".png", is_rgb=True, augmentation=False):
        """
        feat: for save image feature
        root: image path
        key_name: "frames_depth" or other
        list_img_traj: traj_data["images"]. To chose feat[key_name] file
        """
        def _load_with_pt():
            frames_depth = torch.load(os.path.join(root, name_pt)).to(dtype=torch.float32)
            return frames_depth
        path_feat_pt = os.path.join(os.getcwd(), root, name_pt)
        if os.path.isfile(path_feat_pt):
            frames_depth = _load_with_pt()
        else:
            print("{} doesn't exist: {}".format(name_pt, path_feat_pt))
            raise
        # pdb.set_trace()
        frames_depth = self.transform_video_style(frames_depth, augmentation)
        # feat["frames_depth"]
        feat[key_name].append(frames_depth)

    def transform_video_style(self, frames, augmentation=False):
        if augmentation:
            frames = [trans_color(frame) for frame in frames]
        frames = [trans_normalize(frame) for frame in frames]
        frames = torch.stack(frames, dim=0)
        return frames

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.r_state = {
            'state_t': None,
            'e_t': None,
            'cont_lang': None,
            'enc_lang': None
        }

    def step(self, feat, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''

        # encode language features
        if self.r_state['cont_lang'] is None and self.r_state['enc_lang'] is None:
            self.r_state['cont_lang'], self.r_state['enc_lang'] = self.encode_lang(feat)

        # initialize embedding and hidden states
        if self.r_state['e_t'] is None and self.r_state['state_t'] is None:
            self.r_state['e_t'] = self.dec.go.repeat(self.r_state['enc_lang'].size(0), 1)
            self.r_state['state_t'] = self.r_state['cont_lang'], torch.zeros_like(
                self.r_state['cont_lang'])

        # previous action embedding
        e_t = self.embed_action(prev_action) if prev_action is not None else self.r_state['e_t']
        """
        resnet feature
        """
        frames = self.vis_dropout(feat['frames'])
        list_feature_visal = [self.visual_encoder(frame, self.device) for frame in frames]
        feature_visal = torch.stack(list_feature_visal)

        gcn_embedding = self.gcn(feat['frames'])
        # decode and save embedding and hidden states
        out_action_low, out_action_low_mask, state_t, * \
            _ = self.dec.step(self.r_state['enc_lang'], feature_visal
                              [:, 0], e_t=e_t, state_tm1=self.r_state['state_t'], gcn_embedding=gcn_embedding)

        # save states
        self.r_state['state_t'] = state_t
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

            # sigmoid preds to binary mask
            alow_mask = F.sigmoid(alow_mask)
            p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]
            # import pdb; pdb.set_trace()

            task_id_ann = self.get_task_and_ann_id(ex)
            pred[task_id_ann] = {
                'action_low': ' '.join(words),
                'action_low_mask': p_mask,
            }

        return pred

    def embed_action(self, action):
        '''
        embed low-level action
        '''
        device = torch.device("cuda:%d" %
                              self.args.gpu_id) if self.args.gpu else torch.device('cpu')
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
        # [2, 61, 1, 300, 300]
        p_alow_mask = out['out_action_low_mask']
        valid = feat['action_low_valid_interact']

        # action loss
        pad_valid = (l_alow != self.pad)
        alow_loss = F.cross_entropy(p_alow, l_alow, reduction='none')
        alow_loss *= pad_valid.float()
        alow_loss = alow_loss.mean()
        losses['action_low'] = alow_loss * self.args.action_loss_wt

        # mask loss
        valid_idxs = valid.view(-1).nonzero().view(-1)
        flat_p_alow_mask = p_alow_mask.view(
            p_alow_mask.shape[0]*p_alow_mask.shape[1], *p_alow_mask.shape[2:])[valid_idxs]
        flat_alow_mask = torch.cat(feat['action_low_mask'], dim=0)
        alow_mask_loss = self.weighted_mask_loss(flat_p_alow_mask, flat_alow_mask)
        losses['action_low_mask'] = alow_mask_loss * self.args.mask_loss_wt

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
