import torch
from torch import nn
from torch.nn import functional as F
import cv2


class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid, DataParallelDevice=None):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)
        if DataParallelDevice:
            self.scorer = torch.nn.DataParallel(self.scorer, device_ids=DataParallelDevice)

    def forward(self, inp):
        # inp : [2, 145, 1024]
        # [2, 145, 1]
        scores = F.softmax(self.scorer(inp), dim=1)
        # [2, 1, 145] -> [2, 1024]
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        # [2, 145, 1] -> [2, 145, 1024] -> * inp -> sum all 145 word to 1024 feature
        # -> [2, 1024]
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        '''
        inp : [2, 145, 1024]
        h : [2, 1024]
        '''
        # import pdb; pdb.set_trace()
        # [2, 145, 1]
        raw_score = inp.bmm(h.unsqueeze(2))
        # [2, 145, 1]
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe, DataParallelDevice=None):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)
        if DataParallelDevice:
            self.conv1 = torch.nn.DataParallel(self.conv1, device_ids=DataParallelDevice)
            self.conv2 = torch.nn.DataParallel(self.conv2, device_ids=DataParallelDevice)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class VisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super().__init__()
        self.dframe = dframe
        self.flattened_size = 32*36*36

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        # [2, 60, 1, 300, 300]
        x = x.unsqueeze(2)
        len_batch = x.shape[0]
        len_traj = x.shape[1]
        # [120, 1, 300, 300]
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        # [2, 60, 2500]
        x = x.view(len_batch, len_traj, x.shape[1])
        return x


class MaskDecoder(nn.Module):
    '''
    mask decoder
    '''

    def __init__(self, dhid, pframe=300, hshape=(64,7,7)):
        super(MaskDecoder, self).__init__()
        self.dhid = dhid
        self.hshape = hshape
        self.pframe = pframe

        self.d1 = nn.Linear(self.dhid, hshape[0]*hshape[1]*hshape[2])
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(16)
        self.dconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        # [2, 5160]
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        x = self.upsample(x)
        x = self.dconv3(x)
        # [2, 32, 28, 28]
        x = F.relu(self.bn2(x))
        # import pdb; pdb.set_trace()

        x = self.upsample(x)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))
        # import pdb; pdb.set_trace()

        # [2, 1, 300, 300]
        x = self.dconv1(x)
        x = F.interpolate(x, size=(self.pframe, self.pframe), mode='bilinear')
        
        # watch gray
        # import pdb; pdb.set_trace()
        # nx = x[0,0].to(device="cpu").detach().numpy()
        # nx = cv2.cvtColor(nx, cv2.COLOR_GRAY2RGB)
        # cv2.imshow("mask", nx)
        # key = cv2.waitKey(1)
        return x


class ConvFrameMaskDecoder(nn.Module):
    '''
    action decoder
    '''

    def __init__(self, emb, dframe, dhid, dgcn, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb+dgcn, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb+dgcn, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb+dgcn, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, gcn_embedding):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))

        # concat visual feats, weight lang, and previous action embedding
        if gcn_embedding is not None:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, gcn_embedding], dim=1)
        else:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t = state_t[0]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        mask_t = self.mask_dec(cont_t)

        return action_t, mask_t, state_t, lang_attn_t

    def forward(self, enc, frames, gcn_embedding, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t = self.step(enc, frames[:, t], e_t, state_t, gcn_embedding)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'state_t': state_t
        }
        return results


class ConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, dgcn, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        # (input[i], (hx, cx))
        self.cell = nn.LSTMCell(dhid+dframe+demb+dgcn, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb+dgcn, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb+dgcn, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+dgcn, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+dgcn, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, gcn_embedding):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        # import pdb; pdb.set_trace()
        vis_feat_t = self.vis_encoder(frame)
        # [2, 145, 1024]
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        # [2, 1024], [2, 145, 1]
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))
        # import pdb; pdb.set_trace()
        # concat visual feats, weight lang, and previous action embedding
        if gcn_embedding is not None:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, gcn_embedding], dim=1)
        else:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        # [4, 5160]
        cont_t = torch.cat([h_t, inp_t], dim=1)
        # import pdb; pdb.set_trace()
        # [4, 100]
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        # [4, 15]
        action_t = action_emb_t.mm(self.emb.weight.t())

        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t))
        progress_t = torch.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gcn_embedding, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0
        s_gcn_embedding = None
        
        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            if gcn_embedding is not None:
                s_gcn_embedding = gcn_embedding[:, t]
            action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(enc, frames[:, t], e_t, state_t, s_gcn_embedding)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)
        # import pdb; pdb.set_trace()

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results


class DepthConvFrameMaskDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, dgcn, dframedepth, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.vis_depth_encoder = ResnetVisualEncoder(dframe=dframedepth)
        self.cell = nn.LSTMCell(dhid+dframe+demb+dgcn+dframedepth, dhid)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb+dgcn+dframedepth, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, gcn_embedding, frames_depth):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        vis_depth_feat_t = self.vis_depth_encoder(frames_depth)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        # [2, 1024], [2, 145, 1]
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))
        # import pdb; pdb.set_trace()
        # concat visual feats, weight lang, and previous action embedding
        if gcn_embedding is not None:
            gcn_embedding = gcn_embedding.squeeze(0)
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, gcn_embedding, vis_depth_feat_t], dim=1)
        else:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, vis_depth_feat_t], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())

        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t))
        progress_t = torch.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gcn_embedding, frames_depth, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            if gcn_embedding is not None:
                action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(enc, frames[:, t], e_t, state_t, gcn_embedding[:, t], frames_depth[:, t])
            else:
                action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(enc, frames[:, t], e_t, state_t, None, frames_depth[:, t])
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)
        # import pdb; pdb.set_trace()

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results


class DepthAndMemory(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, dgcn, dframedepth,
                 semantic_graph_implement, SEMANTIC_GRAPH_RESULT_FEATURE,
                 pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False,
                 DataParallelDevice=None):
        super().__init__()
        demb = emb.weight.size(1)

        self.semantic_graph_implement = semantic_graph_implement

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe, DataParallelDevice=DataParallelDevice)
        self.vis_depth_encoder = ResnetVisualEncoder(dframe=dframedepth, DataParallelDevice=DataParallelDevice)
        self.attn = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb+dgcn+dframedepth+SEMANTIC_GRAPH_RESULT_FEATURE, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth+SEMANTIC_GRAPH_RESULT_FEATURE, 1)
        self.subgoal = torch.nn.DataParallel(self.subgoal, device_ids=DataParallelDevice)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth+SEMANTIC_GRAPH_RESULT_FEATURE, 1)
        self.progress = torch.nn.DataParallel(self.progress, device_ids=DataParallelDevice)

        self.cell = nn.LSTMCell(dhid+dframe+demb+dgcn+dframedepth+SEMANTIC_GRAPH_RESULT_FEATURE, dhid)
        # self.cell = torch.nn.DataParallel(self.cell, device_ids=DataParallelDevice)
        self.actor = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth+SEMANTIC_GRAPH_RESULT_FEATURE, demb)
        self.actor = torch.nn.DataParallel(self.actor, device_ids=DataParallelDevice)
        self.h_tm1_fc = nn.Linear(dhid, dhid)
        self.h_tm1_fc = torch.nn.DataParallel(self.h_tm1_fc, device_ids=DataParallelDevice)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, gcn_embedding, frames_depth, feat_semantic_graph):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        vis_depth_feat_t = self.vis_depth_encoder(frames_depth)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        # [2, 1024], [2, 145, 1]
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))
        # import pdb; pdb.set_trace()
        # concat visual feats, weight lang, and previous action embedding
        if gcn_embedding is not None:
            gcn_embedding = gcn_embedding.squeeze(0)
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, gcn_embedding, vis_depth_feat_t, feat_semantic_graph], dim=1)
        else:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, vis_depth_feat_t, feat_semantic_graph], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())
        action_t = torch.sigmoid(action_t)

        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t))
        progress_t = torch.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gcn_embedding, frames_depth, all_meta_datas, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            feat_semantic_graph = []
            for env_index in range(len(all_meta_datas)):
                b_store_state = all_meta_datas[env_index]
                if len(b_store_state["sgg_meta_data"]) > t:
                    t_store_state = b_store_state["sgg_meta_data"][t]
                    if t == 0 and self.semantic_graph_implement.use_exploration_frame_feats:
                        exploration_transition_cache = b_store_state["exploration_sgg_meta_data"]
                        self.semantic_graph_implement.update_exploration_data_to_global_graph(
                            exploration_transition_cache,
                            env_index
                        )
                    graph_embed_features, _, _ = \
                        self.semantic_graph_implement.extract_visual_features(
                            store_state=t_store_state,
                            hidden_state=state_t[0][env_index:env_index+1],
                            env_index=env_index
                        )
                else:
                    graph_embed_features = [torch.zeros(
                        1, self.semantic_graph_implement.RESULT_FEATURE).to(frames.device)]
                # graph_embed_features is list (actually dont need list)
                feat_semantic_graph.append(graph_embed_features[0])
            feat_semantic_graph = torch.cat(feat_semantic_graph, dim=0)
            if gcn_embedding is not None:
                action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(
                    enc, frames[:, t], e_t, state_t, gcn_embedding[:, t], frames_depth[:, t], feat_semantic_graph)
            else:
                action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(
                    enc, frames[:, t], e_t, state_t, None, frames_depth[:, t], feat_semantic_graph)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)
        # import pdb; pdb.set_trace()

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results


class DepthConvFrameMaskAttentionDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, dgcn, dframedepth,
                 pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb+dgcn+dframedepth, dhid)
        self.attn = DotAttn()
        self.attn_graph = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb+dgcn+dframedepth, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)
        self.h_tm1_fc_graph = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+dgcn+dframedepth, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, gcn_embedding, frames_depth):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        vis_feat_t = self.vis_encoder(frame)
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        # [2, 1024], [2, 145, 1]
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))
        # concat visual feats, weight lang, and previous action embedding
        if gcn_embedding is not None:
            gcn_embedding, graph_attn_t = self.attn_graph(self.attn_dropout(gcn_embedding), self.h_tm1_fc_graph(h_tm1))
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, gcn_embedding, frames_depth], dim=1)
        else:
            inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, frames_depth], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())

        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t))
        progress_t = torch.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gcn_embedding, frames_depth, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(enc, frames[:, t], e_t, state_t, gcn_embedding, frames_depth[:, t])
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)
        # import pdb; pdb.set_trace()

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results


class ConvFrameMaskAttentionDecoderProgressMonitor(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, dgcn, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False, visual_encode=True, DataParallelDevice=None):
        super().__init__()
        demb = emb.weight.size(1)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        self.visual_encode = visual_encode
        if self.visual_encode:
            self.vis_encoder = ResnetVisualEncoder(dframe=dframe)
        self.cell = nn.LSTMCell(dhid+dframe+demb+dgcn, dhid)
        self.attn = DotAttn()
        self.attn_graph = DotAttn()
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        self.actor = nn.Linear(dhid+dhid+dframe+demb+dgcn, demb)
        self.mask_dec = MaskDecoder(dhid=dhid+dhid+dframe+demb+dgcn, pframe=self.pframe)
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc = nn.Linear(dhid, dhid)
        self.h_tm1_fc_graph = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+dgcn, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+dgcn, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

    def step(self, enc, frame, e_t, state_tm1, gcn_embedding):
        # previous decoder hidden state
        h_tm1 = state_tm1[0]

        # encode vision and lang feat
        if self.visual_encode:
            vis_feat_t = self.vis_encoder(frame)
        else:
            vis_feat_t = frame
        lang_feat_t = enc # language is encoded once at the start

        # attend over language
        # [2, 1024], [2, 145, 1]
        weighted_lang_t, lang_attn_t = self.attn(self.attn_dropout(lang_feat_t), self.h_tm1_fc(h_tm1))
        # concat visual feats, weight lang, and previous action embedding
        gcn_embedding, graph_attn_t = self.attn_graph(self.attn_dropout(gcn_embedding), self.h_tm1_fc_graph(h_tm1))
        inp_t = torch.cat([vis_feat_t, weighted_lang_t, e_t, gcn_embedding], dim=1)
        inp_t = self.input_dropout(inp_t)

        # update hidden state
        state_t = self.cell(inp_t, state_tm1)
        state_t = [self.hstate_dropout(x) for x in state_t]
        h_t, c_t = state_t[0], state_t[1]

        # decode action and mask
        cont_t = torch.cat([h_t, inp_t], dim=1)
        action_emb_t = self.actor(self.actor_dropout(cont_t))
        action_t = action_emb_t.mm(self.emb.weight.t())

        mask_t = self.mask_dec(cont_t)

        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t))
        progress_t = torch.sigmoid(self.progress(cont_t))

        return action_t, mask_t, state_t, lang_attn_t, subgoal_t, progress_t

    def forward(self, enc, frames, gcn_embedding, gold=None, max_decode=150, state_0=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t = state_0

        actions = []
        masks = []
        attn_scores = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            action_t, mask_t, state_t, attn_score_t, subgoal_t, progress_t = self.step(enc, frames[:, t], e_t, state_t, gcn_embedding)
            masks.append(mask_t)
            actions.append(action_t)
            attn_scores.append(attn_score_t)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)

            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                w_t = action_t.max(1)[1]
            e_t = self.emb(w_t)
        # import pdb; pdb.set_trace()

        results = {
            'out_action_low': torch.stack(actions, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_attn_scores': torch.stack(attn_scores, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t': state_t
        }
        return results


class DataParallelDecoder(ConvFrameMaskAttentionDecoderProgressMonitor):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, dframe, dhid, dgcn, pframe=300,
                 attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False, visual_encode=True, DataParallelDevice=None):
        super().__init__(emb, dframe, dhid, dgcn, pframe, attn_dropout, hstate_dropout, actor_dropout, input_dropout, teacher_forcing, visual_encode)
        self.cell = torch.nn.DataParallel(self.cell, device_ids=DataParallelDevice)
        self.actor = torch.nn.DataParallel(self.actor, device_ids=DataParallelDevice)
        self.h_tm1_fc = torch.nn.DataParallel(self.h_tm1_fc, device_ids=DataParallelDevice)
        self.h_tm1_fc_graph = torch.nn.DataParallel(self.h_tm1_fc_graph, device_ids=DataParallelDevice)
