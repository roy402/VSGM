import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from icecream import ic


class SelfAttn(nn.Module):
    '''
    self-attention with learnable parameters
    '''

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)

    def forward(self, inp):
        scores = F.softmax(self.scorer(inp), dim=1)
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        return cont


class DotAttn(nn.Module):
    '''
    dot-attention (or soft-attention)
    '''

    def forward(self, inp, h):
        score = self.softmax(inp, h)
        return score.expand_as(inp).mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        score = F.softmax(raw_score, dim=1)
        return score


class ResnetVisualEncoder(nn.Module):
    '''
    visual encoder
    '''

    def __init__(self, dframe):
        super(ResnetVisualEncoder, self).__init__()
        self.dframe = dframe
        self.flattened_size = 64*7*7

        self.conv1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(self.flattened_size, self.dframe)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = x.view(-1, self.flattened_size)
        x = self.fc(x)

        return x


class AttentionConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(AttentionConv2d, self).__init__()
        self.maskConv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, input):
        attention = self.maskConv(input)
        attention = F.relu(self.bn(attention), inplace=True)
        x = input * attention

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
        self.attentions = AttentionConv2d(16, 16, kernel_size=1)
        self.dconv1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.d1(x))
        x = x.view(-1, *self.hshape)

        x = self.upsample(x)
        x = self.dconv3(x)
        x = F.relu(self.bn2(x))

        x = self.upsample(x)
        x = self.dconv2(x)
        x = F.relu(self.bn1(x))
        x = self.attentions(x)

        x = self.dconv1(x)
        # https://github.com/GuYuc/WS-DAN.PyTorch/blob/87779124f619ceeb445ddfb0246c8a22ff324db4/models/inception.py#L374
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, size=(self.pframe, self.pframe))

        return x


# Thanks to the released code by Federico Landi et al.,
#  dynamic filters are easily exploited.
# see the repo below for more details of dynamic convolution.
#  https://github.com/aimagelab/DynamicConv-agent
######################################################################################################################
class ScaledDotAttn(nn.Module):
    def __init__(self, dim_key_in=1024, dim_key_out=128, dim_query_in=1024 ,dim_query_out=128):
        super().__init__()
        self.fc_key = nn.Linear(dim_key_in, dim_key_out)
        self.fc_query = nn.Linear(dim_query_in, dim_query_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, value, h): # key: lang_feat_t_instr, query: h_tm1_instr
        key = F.relu(self.fc_key(value))
        query = F.relu(self.fc_query(h)).unsqueeze(-1)

        scale_1 = np.sqrt(key.shape[-1])
        scaled_dot_product = torch.bmm(key, query) / scale_1
        softmax = self.softmax(scaled_dot_product)
        element_wise_product = value*softmax
        weighted_lang_t_instr = torch.sum(element_wise_product, dim=1)

        return weighted_lang_t_instr, softmax.squeeze(-1)


class DynamicConvLayer(nn.Module):
    def __init__(self, dhid=512, ):
        super().__init__()
        self.head1 = nn.Linear(dhid, 512)
        self.head2 = nn.Linear(dhid, 512)
        self.head3 = nn.Linear(dhid, 512)
        self.filter_activation = nn.Tanh()

    def forward(self, frame, weighted_lang_t_instr):
        """ dynamic convolutional filters """
        df1 = self.head1(weighted_lang_t_instr)
        df2 = self.head2(weighted_lang_t_instr)
        df3 = self.head3(weighted_lang_t_instr)
        # torch.Size([3, 20, 512])
        # print(torch.stack([df1, df2, df3]).shape)
        # torch.Size([20, 3, 512])
        # print(torch.stack([df1, df2, df3]).transpose(0, 1).shape)
        # import pdb; pdb.set_trace()
        dynamic_filters = torch.stack([df1, df2, df3]).transpose(0, 1)
        dynamic_filters = self.filter_activation(dynamic_filters)
        dynamic_filters = F.normalize(dynamic_filters, p=2, dim=-1)

        """ attention map """
        # torch.Size([20, 512, 7, 7])
        # print(frame.shape)
        frame = frame.view(frame.size(0), frame.size(1), -1)
        # torch.Size([20, 512, 49])
        # print(frame.shape)
        # dynamic_filters.shape  torch.Size([20, 3, 512])
        # print("dynamic_filters.shape ", dynamic_filters.shape)
        # frame.transpose(1,2)  torch.Size([20, 49, 512])
        # print("frame.transpose(1,2) ", frame.transpose(1,2).shape)
        # dynamic_filters.transpose(-1, -2)  torch.Size([20, 512, 3])
        # print("dynamic_filters.transpose(-1, -2) ", dynamic_filters.transpose(-1, -2).shape)
        # import pdb; pdb.set_trace()
        scale_2 = np.sqrt(frame.shape[1]) #torch.sqrt(torch.tensor(frame.shape[1], dtype=torch.double))
        attention_map = torch.bmm(frame.transpose(1,2), dynamic_filters.transpose(-1, -2)) / scale_2
        attention_map = attention_map.reshape(attention_map.size(0), -1)

        return attention_map


class DynamicNodeLayer(nn.Module):
    def __init__(self, dhid=512, ):
        super().__init__()
        self.head1 = nn.Linear(dhid, 1)
        self.head2 = nn.Linear(dhid, 1)
        self.head3 = nn.Linear(dhid, 1)
        self.filter_activation = nn.Tanh()

    def forward(self, node_feature, weighted_lang_t_instr):
        """ dynamic convolutional filters """
        df1 = self.head1(weighted_lang_t_instr)
        df2 = self.head2(weighted_lang_t_instr)
        df3 = self.head3(weighted_lang_t_instr)
        dynamic_filters = torch.stack([df1, df2, df3]).transpose(0, 1)
        dynamic_filters = self.filter_activation(dynamic_filters)
        dynamic_filters = F.normalize(dynamic_filters, p=2, dim=-1)

        """ attention map """
        node_feature = node_feature.view(node_feature.size(0), 1, -1)
        # import pdb; pdb.set_trace()
        scale_2 = np.sqrt(node_feature.shape[1]) #torch.sqrt(torch.tensor(node_feature.shape[1], dtype=torch.double))
        attention_map = torch.bmm(node_feature.transpose(1,2), dynamic_filters.transpose(-1, -2)) / scale_2
        attention_map = attention_map.reshape(attention_map.size(0), -1)
        # torch.Size([20, 512, 3])

        return attention_map


class DecomposeDec(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, num_action_navi_or_operation,
                 dframe, dhid, semantic_graph_implement, IMPORTENT_NDOES_FEATURE,
                 pframe=300, attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.semantic_graph_implement = semantic_graph_implement
        # self.IMPORTENT_NDOES_FEATURE must same with semantic_graph_implement output shape
        # torch.zeros(1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
        self.IMPORTENT_NDOES_FEATURE = IMPORTENT_NDOES_FEATURE
        IMPORTENT_NDOES_FEATURE = IMPORTENT_NDOES_FEATURE*3
        print("self.dynamic_node shape be *3", IMPORTENT_NDOES_FEATURE)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        # LSTM
        self.cell_goal = nn.LSTMCell(dhid+dframe+demb+IMPORTENT_NDOES_FEATURE, dhid)
        self.cell_instr = nn.LSTMCell(dhid+dframe+demb+IMPORTENT_NDOES_FEATURE, dhid)
        print("dhid: ", dhid)
        print("dframe: ", dframe)
        print("demb: ", demb)
        print("self.cell_instr: ", dhid+dframe+demb+IMPORTENT_NDOES_FEATURE)
        self.attn = DotAttn()
        # dropout
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        # predict
        self.actor_navi = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE, demb)
        self.actor_oper = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE, demb)
        self.action_navi_or_operation = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE, num_action_navi_or_operation)
        print("self.actor_navi: ", dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE)
        self.mask_dec = MaskDecoder(dhid=dhid+dframe+IMPORTENT_NDOES_FEATURE, pframe=self.pframe)
        self.mask_dec_label = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(),
            nn.Linear(dhid//2, 119)
        )
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc_goal = nn.Linear(dhid, dhid)
        self.h_tm1_fc_instr = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

        self.scale_dot_attn = ScaledDotAttn(dhid, 128, dhid, 128)
        self.dynamic_conv = DynamicConvLayer(dhid)
        self.dynamic_node = DynamicNodeLayer(dhid)

    def step(
            self,
            enc_goal,
            enc_instr,
            frame,
            e_t,
            state_tm1_goal,
            state_tm1_instr,
            feat_global_graph,
            feat_current_state_graph,
            feat_history_changed_nodes_graph,
            feat_priori_graph):
        # previous decoder hidden state (goal, instr decoder)
        h_tm1_goal = state_tm1_goal[0]
        h_tm1_instr = state_tm1_instr[0]

        # encode vision and lang feat (goal, instr decoder)
        lang_feat_t_goal = enc_goal # language is encoded once at the start
        lang_feat_t_instr = enc_instr # language is encoded once at the start

        # scaled dot product attention
        weighted_lang_t_goal, lang_attn_t_goal = self.scale_dot_attn(lang_feat_t_goal, h_tm1_goal)
        weighted_lang_t_instr, lang_attn_t_instr = self.scale_dot_attn(lang_feat_t_instr, h_tm1_instr)

        # dynamic convolution
        vis_feat_t_goal = self.dynamic_conv(frame, weighted_lang_t_instr)
        vis_feat_t_instr = self.dynamic_conv(frame, weighted_lang_t_instr)
        feat_global_graph = self.dynamic_node(feat_global_graph, weighted_lang_t_instr)
        feat_current_state_graph = self.dynamic_node(feat_current_state_graph, weighted_lang_t_instr)
        feat_history_changed_nodes_graph = self.dynamic_node(feat_history_changed_nodes_graph, weighted_lang_t_instr)
        feat_priori_graph = self.dynamic_node(feat_priori_graph, weighted_lang_t_instr)

        # concat visual feats, weight lang, and previous action embedding (goal decoder)
        inp_t_goal = torch.cat([vis_feat_t_goal, weighted_lang_t_goal, e_t, feat_priori_graph], dim=1)
        inp_t_goal = self.input_dropout(inp_t_goal)

        # concat visual feats, weight lang, and previous action embedding (instr decoder)
        inp_t_instr = torch.cat([vis_feat_t_instr, weighted_lang_t_instr, e_t, feat_current_state_graph], dim=1)
        inp_t_instr = self.input_dropout(inp_t_instr)

        # update hidden state (goal decoder)
        state_t_goal = self.cell_goal(inp_t_goal, state_tm1_goal)
        state_t_goal = [self.hstate_dropout(x) for x in state_t_goal]
        h_t_goal, _ = state_t_goal[0], state_t_goal[1]

        # decode mask (goal decoder)
        cont_t_goal = h_t_goal #torch.cat([h_t_goal, inp_t_goal], dim=1)
        masks_label_t = self.mask_dec_label(cont_t_goal)

        # update hidden state (instr decoder)
        state_t_instr = self.cell_instr(inp_t_instr, state_tm1_instr)
        state_t_instr = [self.hstate_dropout(x) for x in state_t_instr]
        h_t_instr, _ = state_t_instr[0], state_t_instr[1]

        mask_input = torch.cat([h_t_instr, vis_feat_t_instr, feat_priori_graph], dim=1)
        mask_input = self.input_dropout(mask_input)
        mask_t = self.mask_dec(mask_input)

        # decode action (instr decoder)
        cont_t_instr = torch.cat([h_t_instr, inp_t_instr], dim=1)
        action_navi_emb_t = self.actor_navi(self.actor_dropout(cont_t_instr))
        action_oper_emb_t = self.actor_oper(self.actor_dropout(cont_t_instr))
        action_navi_or_operation_t = self.action_navi_or_operation(self.actor_dropout(cont_t_instr))
        action_navi_t = action_navi_emb_t.mm(self.emb.weight.t())
        action_oper_t = action_oper_emb_t.mm(self.emb.weight.t())

        cont_t_instr_with_subgoal = torch.cat(
            [h_t_instr, vis_feat_t_instr, weighted_lang_t_instr, e_t,
                feat_history_changed_nodes_graph], dim=1)
        cont_t_instr_with_progress = torch.cat(
            [h_t_instr, vis_feat_t_instr, weighted_lang_t_instr, e_t,
                feat_history_changed_nodes_graph, feat_global_graph], dim=1)
        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t_instr_with_subgoal))
        progress_t = torch.sigmoid(self.progress(cont_t_instr_with_progress))
        action_navi_or_operation_t = torch.sigmoid(action_navi_or_operation_t)

        return action_navi_t, action_oper_t, mask_t, action_navi_or_operation_t, masks_label_t,\
            state_t_goal, state_t_instr, lang_attn_t_goal, lang_attn_t_instr, subgoal_t, progress_t,\
            weighted_lang_t_goal, weighted_lang_t_instr

    def forward(self, enc_goal, enc_instr, frames, all_meta_datas, gold=None, max_decode=150, state_0_goal=None, state_0_instr=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc_instr.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t_goal = state_0_goal
        state_t_instr = state_0_instr

        actions_navi, actions_oper, actions_navi_or_operation = [], [], []
        masks, masks_label = [], []
        attn_scores_goal = []
        attn_scores_instr = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            feat_global_graph = []
            feat_current_state_graph = []
            feat_history_changed_nodes_graph = []
            feat_priori_graph = []
            for env_index in range(len(all_meta_datas)):
                b_store_state = all_meta_datas[env_index]
                if len(b_store_state["sgg_meta_data"]) > t:
                    t_store_state = b_store_state["sgg_meta_data"][t]
                    if t == 0 and self.semantic_graph_implement.use_exploration_frame_feats:
                        # import pdb;pdb.set_trace()
                        exploration_transition_cache = b_store_state["exploration_sgg_meta_data"]
                        exploration_imgs = b_store_state["exploration_imgs"]
                        self.semantic_graph_implement.update_exploration_data_to_global_graph(
                            exploration_transition_cache,
                            env_index,
                            exploration_imgs=exploration_imgs,
                        )
                    t_store_state["rgb_image"] = frames[env_index, t]
                    # import pdb; pdb.set_trace()
                    global_graph_importent_features, current_state_graph_importent_features, history_changed_nodes_graph_importent_features, priori_importent_features, *_ =\
                        self.store_and_get_graph_feature(t_store_state, env_index, state_t_goal, state_t_instr)
                else:
                    global_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    current_state_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    history_changed_nodes_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    priori_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                feat_global_graph.append(global_graph_importent_features)
                feat_current_state_graph.append(current_state_graph_importent_features)
                feat_history_changed_nodes_graph.append(history_changed_nodes_graph_importent_features)
                feat_priori_graph.append(priori_importent_features)
            feat_global_graph = torch.cat(feat_global_graph, dim=0)
            feat_current_state_graph = torch.cat(feat_current_state_graph, dim=0)
            feat_history_changed_nodes_graph = torch.cat(feat_history_changed_nodes_graph, dim=0)
            feat_priori_graph = torch.cat(feat_priori_graph, dim=0)

            action_navi_t, action_oper_t, mask_t, action_navi_or_operation_t, masks_label_t,\
                state_t_goal, state_t_instr, attn_score_t_goal, attn_score_t_instr, subgoal_t, progress_t,\
                weighted_lang_t_goal, weighted_lang_t_instr = \
                self.step(
                    enc_goal,
                    enc_instr,
                    frames[:, t],
                    e_t,
                    state_t_goal,
                    state_t_instr,
                    feat_global_graph,
                    feat_current_state_graph,
                    feat_history_changed_nodes_graph,
                    feat_priori_graph)
            masks.append(mask_t)
            masks_label.append(masks_label_t)
            actions_navi.append(action_navi_t)
            actions_oper.append(action_oper_t)
            actions_navi_or_operation.append(action_navi_or_operation_t)
            attn_scores_goal.append(attn_score_t_goal)
            attn_scores_instr.append(attn_score_t_instr)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)
            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                # w_t = action_t.max(1)[1]
                w_t = self.chose_embed_index(action_navi_t, action_oper_t, action_navi_or_operation_t)

            e_t = self.emb(w_t)
        results = {
            'out_action_navi_low': torch.stack(actions_navi, dim=1),
            'out_action_operation_low': torch.stack(actions_oper, dim=1),
            'out_action_navi_or_operation': torch.stack(actions_navi_or_operation, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_action_low_mask_label': torch.stack(masks_label, dim=1),
            'out_attn_scores_goal': torch.stack(attn_scores_goal, dim=1),
            'out_attn_scores_instr': torch.stack(attn_scores_instr, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t_goal': state_t_goal,
            'state_t_instr': state_t_instr,
        }
        return results

    def store_and_get_graph_feature(self, t_store_state, env_index, state_t_goal, state_t_instr):
        self.semantic_graph_implement.store_data_to_graph(
            store_state=t_store_state,
            env_index=env_index
        )
        global_graph_importent_features, global_graph_dict_objectIds_to_score = \
            self.semantic_graph_implement.get_graph_feature(
                chose_type="GLOBAL_GRAPH",
                env_index=env_index,
                )
        current_state_graph_importent_features, current_state_dict_objectIds_to_score = \
            self.semantic_graph_implement.chose_importent_node_feature(
                chose_type="CURRENT_STATE_GRAPH",
                env_index=env_index,
                hidden_state=state_t_instr[0][env_index:env_index+1],
                )
        history_changed_nodes_graph_importent_features, history_changed_dict_objectIds_to_score = \
            self.semantic_graph_implement.chose_importent_node_feature(
                chose_type="HISTORY_CHANGED_NODES_GRAPH",
                env_index=env_index,
                hidden_state=state_t_goal[0][env_index:env_index+1],
                )
        priori_importent_features, priori_dict_dict_objectIds_to_score = \
            self.semantic_graph_implement.chose_importent_node_feature(
                chose_type="PRIORI_GRAPH",
                env_index=env_index,
                hidden_state=state_t_instr[0][env_index:env_index+1],
                )
        return global_graph_importent_features, current_state_graph_importent_features,\
               history_changed_nodes_graph_importent_features, priori_importent_features,\
               global_graph_dict_objectIds_to_score, current_state_dict_objectIds_to_score,\
               history_changed_dict_objectIds_to_score, priori_dict_dict_objectIds_to_score

    # for model predict use Detailed Operation model || Navi model result
    def chose_embed_index(self, action_navi_t, action_oper_t, action_navi_or_operation_t):
        '''
        action_navi_t: embed index, [0~x]
        '''
        navi_or_operation_t = action_navi_or_operation_t.max(1)[1]
        w_t_navi = action_navi_t.max(1)[1]
        w_t_oper = action_oper_t.max(1)[1]
        w_t = [w_t_oper[i] if is_o else w_t_navi[i] for i, is_o in enumerate(navi_or_operation_t)]
        w_t = torch.stack(w_t)
        return w_t


class DecomposeDec2(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, num_action_navi_or_operation,
                 dframe, dhid, semantic_graph_implement, IMPORTENT_NDOES_FEATURE,
                 pframe=300, attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.semantic_graph_implement = semantic_graph_implement
        # self.IMPORTENT_NDOES_FEATURE must same with semantic_graph_implement output shape
        # torch.zeros(1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
        self.IMPORTENT_NDOES_FEATURE = IMPORTENT_NDOES_FEATURE
        IMPORTENT_NDOES_FEATURE = IMPORTENT_NDOES_FEATURE*3
        print("self.dynamic_node shape be *3", IMPORTENT_NDOES_FEATURE)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        # LSTM
        self.cell_goal = nn.LSTMCell(dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, dhid)
        self.cell_instr = nn.LSTMCell(dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, dhid)
        print("dhid: ", dhid)
        print("dframe: ", dframe)
        print("demb: ", demb)
        print("self.cell_instr: ", dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE)
        self.attn = DotAttn()
        # dropout
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        # predict
        self.actor_navi = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, demb)
        self.actor_oper = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, demb)
        self.action_navi_or_operation = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, num_action_navi_or_operation)
        print("self.actor_navi: ", dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE)
        self.mask_dec = MaskDecoder(dhid=dhid+dframe, pframe=self.pframe)
        self.mask_dec_label = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(),
            nn.Linear(dhid//2, 119)
        )
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc_goal = nn.Linear(dhid, dhid)
        self.h_tm1_fc_instr = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

        self.scale_dot_attn = ScaledDotAttn(dhid, 128, dhid, 128)
        self.dynamic_conv = DynamicConvLayer(dhid)
        self.dynamic_node = DynamicNodeLayer(dhid)

    def step(
            self,
            enc_goal,
            enc_instr,
            frame,
            e_t,
            state_tm1_goal,
            state_tm1_instr,
            feat_global_graph,
            feat_current_state_graph,
            feat_history_changed_nodes_graph,
            feat_priori_graph):
        # previous decoder hidden state (goal, instr decoder)
        h_tm1_goal = state_tm1_goal[0]
        h_tm1_instr = state_tm1_instr[0]

        # encode vision and lang feat (goal, instr decoder)
        lang_feat_t_goal = enc_goal # language is encoded once at the start
        lang_feat_t_instr = enc_instr # language is encoded once at the start

        # scaled dot product attention
        weighted_lang_t_goal, lang_attn_t_goal = self.scale_dot_attn(lang_feat_t_goal, h_tm1_goal)
        weighted_lang_t_instr, lang_attn_t_instr = self.scale_dot_attn(lang_feat_t_instr, h_tm1_instr)

        # dynamic convolution
        vis_feat_t_goal = self.dynamic_conv(frame, weighted_lang_t_instr)
        vis_feat_t_instr = self.dynamic_conv(frame, weighted_lang_t_instr)
        feat_global_graph = self.dynamic_node(feat_global_graph, weighted_lang_t_instr)
        feat_current_state_graph = self.dynamic_node(feat_current_state_graph, weighted_lang_t_instr)
        feat_history_changed_nodes_graph = self.dynamic_node(feat_history_changed_nodes_graph, weighted_lang_t_instr)
        feat_priori_graph = self.dynamic_node(feat_priori_graph, weighted_lang_t_instr)

        # concat visual feats, weight lang, and previous action embedding (goal decoder)
        inp_t_goal = torch.cat([vis_feat_t_goal, weighted_lang_t_goal, e_t, feat_priori_graph, feat_global_graph], dim=1)
        inp_t_goal = self.input_dropout(inp_t_goal)

        # concat visual feats, weight lang, and previous action embedding (instr decoder)
        inp_t_instr = torch.cat([vis_feat_t_instr, weighted_lang_t_instr, e_t, feat_global_graph, feat_current_state_graph], dim=1)
        inp_t_instr = self.input_dropout(inp_t_instr)

        # update hidden state (goal decoder)
        state_t_goal = self.cell_goal(inp_t_goal, state_tm1_goal)
        state_t_goal = [self.hstate_dropout(x) for x in state_t_goal]
        h_t_goal, _ = state_t_goal[0], state_t_goal[1]

        # decode mask (goal decoder)
        cont_t_goal = h_t_goal #torch.cat([h_t_goal, inp_t_goal], dim=1)
        masks_label_t = self.mask_dec_label(cont_t_goal)

        # update hidden state (instr decoder)
        state_t_instr = self.cell_instr(inp_t_instr, state_tm1_instr)
        state_t_instr = [self.hstate_dropout(x) for x in state_t_instr]
        h_t_instr, _ = state_t_instr[0], state_t_instr[1]

        mask_input = torch.cat([h_t_instr, vis_feat_t_instr], dim=1)
        mask_input = self.input_dropout(mask_input)
        mask_t = self.mask_dec(mask_input)

        # decode action (instr decoder)
        cont_t_instr = torch.cat([h_t_instr, inp_t_instr], dim=1)
        action_navi_emb_t = self.actor_navi(self.actor_dropout(cont_t_instr))
        action_oper_emb_t = self.actor_oper(self.actor_dropout(cont_t_instr))
        action_navi_or_operation_t = self.action_navi_or_operation(self.actor_dropout(cont_t_instr))
        action_navi_t = action_navi_emb_t.mm(self.emb.weight.t())
        action_oper_t = action_oper_emb_t.mm(self.emb.weight.t())

        cont_t_instr_with_subgoal = torch.cat(
            [h_t_instr, vis_feat_t_instr, weighted_lang_t_instr, e_t,
                feat_history_changed_nodes_graph, feat_global_graph], dim=1)
        cont_t_instr_with_progress = torch.cat(
            [h_t_instr, vis_feat_t_instr, weighted_lang_t_instr, e_t,
                feat_history_changed_nodes_graph, feat_global_graph], dim=1)
        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t_instr_with_subgoal))
        progress_t = torch.sigmoid(self.progress(cont_t_instr_with_progress))
        action_navi_or_operation_t = torch.sigmoid(action_navi_or_operation_t)

        return action_navi_t, action_oper_t, mask_t, action_navi_or_operation_t, masks_label_t,\
            state_t_goal, state_t_instr, lang_attn_t_goal, lang_attn_t_instr, subgoal_t, progress_t,\
            weighted_lang_t_goal, weighted_lang_t_instr

    def forward(self, enc_goal, enc_instr, frames, all_meta_datas, gold=None, max_decode=150, state_0_goal=None, state_0_instr=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc_instr.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t_goal = state_0_goal
        state_t_instr = state_0_instr

        actions_navi, actions_oper, actions_navi_or_operation = [], [], []
        masks, masks_label = [], []
        attn_scores_goal = []
        attn_scores_instr = []
        subgoals = []
        progresses = []
        for t in range(max_t):
            feat_global_graph = []
            feat_current_state_graph = []
            feat_history_changed_nodes_graph = []
            feat_priori_graph = []
            for env_index in range(len(all_meta_datas)):
                b_store_state = all_meta_datas[env_index]
                if len(b_store_state["sgg_meta_data"]) > t:
                    t_store_state = b_store_state["sgg_meta_data"][t]
                    if t == 0 and self.semantic_graph_implement.use_exploration_frame_feats:
                        # import pdb;pdb.set_trace()
                        exploration_transition_cache = b_store_state["exploration_sgg_meta_data"]
                        exploration_imgs = b_store_state["exploration_imgs"]
                        self.semantic_graph_implement.update_exploration_data_to_global_graph(
                            exploration_transition_cache,
                            env_index,
                            exploration_imgs=exploration_imgs,
                        )
                    t_store_state["rgb_image"] = frames[env_index, t]
                    # import pdb; pdb.set_trace()
                    global_graph_importent_features, current_state_graph_importent_features, history_changed_nodes_graph_importent_features, priori_importent_features, *_ =\
                        self.store_and_get_graph_feature(t_store_state, env_index, state_t_goal, state_t_instr)
                else:
                    global_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    current_state_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    history_changed_nodes_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    priori_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                feat_global_graph.append(global_graph_importent_features)
                feat_current_state_graph.append(current_state_graph_importent_features)
                feat_history_changed_nodes_graph.append(history_changed_nodes_graph_importent_features)
                feat_priori_graph.append(priori_importent_features)
            feat_global_graph = torch.cat(feat_global_graph, dim=0)
            feat_current_state_graph = torch.cat(feat_current_state_graph, dim=0)
            feat_history_changed_nodes_graph = torch.cat(feat_history_changed_nodes_graph, dim=0)
            feat_priori_graph = torch.cat(feat_priori_graph, dim=0)

            action_navi_t, action_oper_t, mask_t, action_navi_or_operation_t, masks_label_t,\
                state_t_goal, state_t_instr, attn_score_t_goal, attn_score_t_instr, subgoal_t, progress_t,\
                weighted_lang_t_goal, weighted_lang_t_instr = \
                self.step(
                    enc_goal,
                    enc_instr,
                    frames[:, t],
                    e_t,
                    state_t_goal,
                    state_t_instr,
                    feat_global_graph,
                    feat_current_state_graph,
                    feat_history_changed_nodes_graph,
                    feat_priori_graph)
            masks.append(mask_t)
            masks_label.append(masks_label_t)
            actions_navi.append(action_navi_t)
            actions_oper.append(action_oper_t)
            actions_navi_or_operation.append(action_navi_or_operation_t)
            attn_scores_goal.append(attn_score_t_goal)
            attn_scores_instr.append(attn_score_t_instr)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)
            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                # w_t = action_t.max(1)[1]
                w_t = self.chose_embed_index(action_navi_t, action_oper_t, action_navi_or_operation_t)

            e_t = self.emb(w_t)
        results = {
            'out_action_navi_low': torch.stack(actions_navi, dim=1),
            'out_action_operation_low': torch.stack(actions_oper, dim=1),
            'out_action_navi_or_operation': torch.stack(actions_navi_or_operation, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_action_low_mask_label': torch.stack(masks_label, dim=1),
            'out_attn_scores_goal': torch.stack(attn_scores_goal, dim=1),
            'out_attn_scores_instr': torch.stack(attn_scores_instr, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t_goal': state_t_goal,
            'state_t_instr': state_t_instr,
        }
        return results

    def store_and_get_graph_feature(self, t_store_state, env_index, state_t_goal, state_t_instr):
        self.semantic_graph_implement.store_data_to_graph(
            store_state=t_store_state,
            env_index=env_index
        )
        global_graph_importent_features, global_graph_dict_objectIds_to_score = \
            self.semantic_graph_implement.chose_importent_node_feature(
                chose_type="GLOBAL_GRAPH",
                env_index=env_index,
                hidden_state=state_t_goal[0][env_index:env_index+1],
                )
        current_state_graph_importent_features, current_state_dict_objectIds_to_score = \
            self.semantic_graph_implement.get_graph_feature(
                chose_type="CURRENT_STATE_GRAPH",
                env_index=env_index,
                )
        history_changed_nodes_graph_importent_features, history_changed_dict_objectIds_to_score = \
            self.semantic_graph_implement.get_graph_feature(
                chose_type="HISTORY_CHANGED_NODES_GRAPH",
                env_index=env_index,
                )
        priori_importent_features, priori_dict_dict_objectIds_to_score = \
            self.semantic_graph_implement.get_graph_feature(
                chose_type="PRIORI_GRAPH",
                env_index=env_index,
                # hidden_state=state_t_instr[0][env_index:env_index+1],
                )
        return global_graph_importent_features, current_state_graph_importent_features,\
               history_changed_nodes_graph_importent_features, priori_importent_features,\
               global_graph_dict_objectIds_to_score, current_state_dict_objectIds_to_score,\
               history_changed_dict_objectIds_to_score, priori_dict_dict_objectIds_to_score

    # for model predict use Detailed Operation model || Navi model result
    def chose_embed_index(self, action_navi_t, action_oper_t, action_navi_or_operation_t):
        '''
        action_navi_t: embed index, [0~x]
        '''
        navi_or_operation_t = action_navi_or_operation_t.max(1)[1]
        w_t_navi = action_navi_t.max(1)[1]
        w_t_oper = action_oper_t.max(1)[1]
        w_t = [w_t_oper[i] if is_o else w_t_navi[i] for i, is_o in enumerate(navi_or_operation_t)]
        w_t = torch.stack(w_t)
        return w_t


class ContrastiveDecompose(nn.Module):
    '''
    action decoder with subgoal and progress monitoring
    '''

    def __init__(self, emb, num_action_navi_or_operation,
                 dframe, dhid, semantic_graph_implement, IMPORTENT_NDOES_FEATURE,
                 pframe=300, attn_dropout=0., hstate_dropout=0., actor_dropout=0., input_dropout=0.,
                 teacher_forcing=False):
        super().__init__()
        demb = emb.weight.size(1)

        self.semantic_graph_implement = semantic_graph_implement
        # self.IMPORTENT_NDOES_FEATURE must same with semantic_graph_implement output shape
        # torch.zeros(1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
        self.IMPORTENT_NDOES_FEATURE = IMPORTENT_NDOES_FEATURE
        IMPORTENT_NDOES_FEATURE = IMPORTENT_NDOES_FEATURE*3
        print("self.dynamic_node shape be *3", IMPORTENT_NDOES_FEATURE)

        self.emb = emb
        self.pframe = pframe
        self.dhid = dhid
        # LSTM
        self.cell_goal = nn.LSTMCell(dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, dhid)
        self.cell_instr = nn.LSTMCell(dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, dhid)
        print("dhid: ", dhid)
        print("dframe: ", dframe)
        print("demb: ", demb)
        print("self.cell_instr: ", dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE)
        self.attn = DotAttn()
        # dropout
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.hstate_dropout = nn.Dropout(hstate_dropout)
        self.actor_dropout = nn.Dropout(actor_dropout)
        self.go = nn.Parameter(torch.Tensor(demb))
        # predict
        self.actor_navi = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, demb)
        self.actor_oper = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, demb)
        self.action_navi_or_operation = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, num_action_navi_or_operation)
        print("self.actor_navi: ", dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE)
        self.mask_dec = MaskDecoder(dhid=dhid+dframe, pframe=self.pframe)
        self.mask_dec_label = nn.Sequential(
            nn.Linear(dhid, dhid//2), nn.ReLU(),
            nn.Linear(dhid//2, 119)
        )
        self.teacher_forcing = teacher_forcing
        self.h_tm1_fc_goal = nn.Linear(dhid, dhid)
        self.h_tm1_fc_instr = nn.Linear(dhid, dhid)

        self.subgoal = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, 1)
        self.progress = nn.Linear(dhid+dhid+dframe+demb+IMPORTENT_NDOES_FEATURE+IMPORTENT_NDOES_FEATURE, 1)

        nn.init.uniform_(self.go, -0.1, 0.1)

        self.scale_dot_attn = ScaledDotAttn(dhid, 128, dhid, 128)
        self.dynamic_conv = DynamicConvLayer(dhid)
        self.dynamic_node = DynamicNodeLayer(dhid)

    def step(
            self,
            enc_goal,
            enc_instr,
            frame,
            e_t,
            state_tm1_goal,
            state_tm1_instr,
            feat_global_graph,
            feat_current_state_graph,
            feat_history_changed_nodes_graph,
            feat_priori_graph):
        # previous decoder hidden state (goal, instr decoder)
        h_tm1_goal = state_tm1_goal[0]
        h_tm1_instr = state_tm1_instr[0]

        # encode vision and lang feat (goal, instr decoder)
        lang_feat_t_goal = enc_goal # language is encoded once at the start
        lang_feat_t_instr = enc_instr # language is encoded once at the start

        # scaled dot product attention
        weighted_lang_t_goal, lang_attn_t_goal = self.scale_dot_attn(lang_feat_t_goal, h_tm1_goal)
        weighted_lang_t_instr, lang_attn_t_instr = self.scale_dot_attn(lang_feat_t_instr, h_tm1_instr)

        # dynamic convolution
        vis_feat_t_goal = self.dynamic_conv(frame, weighted_lang_t_instr)
        vis_feat_t_instr = self.dynamic_conv(frame, weighted_lang_t_instr)
        feat_global_graph = self.dynamic_node(feat_global_graph, weighted_lang_t_instr)
        feat_current_state_graph = self.dynamic_node(feat_current_state_graph, weighted_lang_t_instr)
        feat_history_changed_nodes_graph = self.dynamic_node(feat_history_changed_nodes_graph, weighted_lang_t_instr)
        feat_priori_graph = self.dynamic_node(feat_priori_graph, weighted_lang_t_instr)

        # concat visual feats, weight lang, and previous action embedding (goal decoder)
        inp_t_goal = torch.cat([vis_feat_t_goal, weighted_lang_t_goal, e_t, feat_priori_graph, feat_global_graph], dim=1)
        inp_t_goal = self.input_dropout(inp_t_goal)

        # concat visual feats, weight lang, and previous action embedding (instr decoder)
        inp_t_instr = torch.cat([vis_feat_t_instr, weighted_lang_t_instr, e_t, feat_global_graph, feat_current_state_graph], dim=1)
        inp_t_instr = self.input_dropout(inp_t_instr)

        # update hidden state (goal decoder)
        state_t_goal = self.cell_goal(inp_t_goal, state_tm1_goal)
        state_t_goal = [self.hstate_dropout(x) for x in state_t_goal]
        h_t_goal, _ = state_t_goal[0], state_t_goal[1]

        # decode mask (goal decoder)
        cont_t_goal = h_t_goal #torch.cat([h_t_goal, inp_t_goal], dim=1)
        masks_label_t = self.mask_dec_label(cont_t_goal)

        # update hidden state (instr decoder)
        state_t_instr = self.cell_instr(inp_t_instr, state_tm1_instr)
        state_t_instr = [self.hstate_dropout(x) for x in state_t_instr]
        h_t_instr, _ = state_t_instr[0], state_t_instr[1]

        mask_input = torch.cat([h_t_instr, vis_feat_t_instr], dim=1)
        mask_input = self.input_dropout(mask_input)
        mask_t = self.mask_dec(mask_input)

        # decode action (instr decoder)
        cont_t_instr = torch.cat([h_t_instr, inp_t_instr], dim=1)
        action_navi_emb_t = self.actor_navi(self.actor_dropout(cont_t_instr))
        action_oper_emb_t = self.actor_oper(self.actor_dropout(cont_t_instr))
        action_navi_or_operation_t = self.action_navi_or_operation(self.actor_dropout(cont_t_instr))
        action_navi_t = action_navi_emb_t.mm(self.emb.weight.t())
        action_oper_t = action_oper_emb_t.mm(self.emb.weight.t())

        cont_t_instr_with_subgoal = torch.cat(
            [h_t_instr, vis_feat_t_instr, weighted_lang_t_instr, e_t,
                feat_history_changed_nodes_graph, feat_global_graph], dim=1)
        cont_t_instr_with_progress = torch.cat(
            [h_t_instr, vis_feat_t_instr, weighted_lang_t_instr, e_t,
                feat_history_changed_nodes_graph, feat_global_graph], dim=1)
        # predict subgoals completed and task progress
        subgoal_t = torch.sigmoid(self.subgoal(cont_t_instr_with_subgoal))
        progress_t = torch.sigmoid(self.progress(cont_t_instr_with_progress))
        action_navi_or_operation_t = torch.sigmoid(action_navi_or_operation_t)

        return action_navi_t, action_oper_t, mask_t, action_navi_or_operation_t, masks_label_t,\
            state_t_goal, state_t_instr, lang_attn_t_goal, lang_attn_t_instr, subgoal_t, progress_t,\
            weighted_lang_t_goal, weighted_lang_t_instr

    def forward(self, enc_goal, enc_instr, frames, all_meta_datas, gold=None, max_decode=150, state_0_goal=None, state_0_instr=None):
        max_t = gold.size(1) if self.training else min(max_decode, frames.shape[1])
        batch = enc_instr.size(0)
        e_t = self.go.repeat(batch, 1)
        state_t_goal = state_0_goal
        state_t_instr = state_0_instr

        actions_navi, actions_oper, actions_navi_or_operation = [], [], []
        masks, masks_label = [], []
        attn_scores_goal = []
        attn_scores_instr = []
        subgoals = []
        progresses = []
        contrastive_data = {
            "weighted_lang_t_goal": [],
            "weighted_lang_t_instr": [],
            "state_t_goal": [],
            "state_t_instr": [],
            "global_graph": [],
            "current_graph": [],
            "history_graph": [],
        }
        for t in range(max_t):
            feat_global_graph = []
            feat_current_state_graph = []
            feat_history_changed_nodes_graph = []
            feat_priori_graph = []
            for env_index in range(len(all_meta_datas)):
                b_store_state = all_meta_datas[env_index]
                if len(b_store_state["sgg_meta_data"]) > t:
                    t_store_state = b_store_state["sgg_meta_data"][t]
                    if t == 0 and self.semantic_graph_implement.use_exploration_frame_feats:
                        # import pdb;pdb.set_trace()
                        exploration_transition_cache = b_store_state["exploration_sgg_meta_data"]
                        exploration_imgs = b_store_state["exploration_imgs"]
                        self.semantic_graph_implement.update_exploration_data_to_global_graph(
                            exploration_transition_cache,
                            env_index,
                            exploration_imgs=exploration_imgs,
                        )
                    t_store_state["rgb_image"] = frames[env_index, t]
                    # import pdb; pdb.set_trace()
                    global_graph_importent_features, current_state_graph_importent_features, history_changed_nodes_graph_importent_features, priori_importent_features, *_ =\
                        self.store_and_get_graph_feature(t_store_state, env_index, state_t_goal, state_t_instr)
                else:
                    global_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    current_state_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    history_changed_nodes_graph_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                    priori_importent_features = torch.zeros(
                        1, self.IMPORTENT_NDOES_FEATURE).to(frames.device)
                feat_global_graph.append(global_graph_importent_features)
                feat_current_state_graph.append(current_state_graph_importent_features)
                feat_history_changed_nodes_graph.append(history_changed_nodes_graph_importent_features)
                feat_priori_graph.append(priori_importent_features)
            feat_global_graph = torch.cat(feat_global_graph, dim=0)
            feat_current_state_graph = torch.cat(feat_current_state_graph, dim=0)
            feat_history_changed_nodes_graph = torch.cat(feat_history_changed_nodes_graph, dim=0)
            feat_priori_graph = torch.cat(feat_priori_graph, dim=0)

            action_navi_t, action_oper_t, mask_t, action_navi_or_operation_t, masks_label_t,\
                state_t_goal, state_t_instr, attn_score_t_goal, attn_score_t_instr, subgoal_t, progress_t,\
                weighted_lang_t_goal, weighted_lang_t_instr = \
                self.step(
                    enc_goal,
                    enc_instr,
                    frames[:, t],
                    e_t,
                    state_t_goal,
                    state_t_instr,
                    feat_global_graph,
                    feat_current_state_graph,
                    feat_history_changed_nodes_graph,
                    feat_priori_graph)
            masks.append(mask_t)
            masks_label.append(masks_label_t)
            actions_navi.append(action_navi_t)
            actions_oper.append(action_oper_t)
            actions_navi_or_operation.append(action_navi_or_operation_t)
            attn_scores_goal.append(attn_score_t_goal)
            attn_scores_instr.append(attn_score_t_instr)
            subgoals.append(subgoal_t)
            progresses.append(progress_t)
            contrastive_data["weighted_lang_t_goal"].append(weighted_lang_t_goal)
            contrastive_data["weighted_lang_t_instr"].append(weighted_lang_t_instr)
            contrastive_data["state_t_goal"].append(state_t_goal[0])
            contrastive_data["state_t_instr"].append(state_t_instr[0])
            contrastive_data["global_graph"].append(feat_global_graph)
            contrastive_data["current_graph"].append(feat_current_state_graph)
            contrastive_data["history_graph"].append(feat_history_changed_nodes_graph)
            # find next emb
            if self.teacher_forcing and self.training:
                w_t = gold[:, t]
            else:
                # w_t = action_t.max(1)[1]
                w_t = self.chose_embed_index(action_navi_t, action_oper_t, action_navi_or_operation_t)

            e_t = self.emb(w_t)

        for k, v in contrastive_data.items():
            contrastive_data[k] = torch.stack(v, dim=1)
        results = {
            'out_action_navi_low': torch.stack(actions_navi, dim=1),
            'out_action_operation_low': torch.stack(actions_oper, dim=1),
            'out_action_navi_or_operation': torch.stack(actions_navi_or_operation, dim=1),
            'out_action_low_mask': torch.stack(masks, dim=1),
            'out_action_low_mask_label': torch.stack(masks_label, dim=1),
            'out_attn_scores_goal': torch.stack(attn_scores_goal, dim=1),
            'out_attn_scores_instr': torch.stack(attn_scores_instr, dim=1),
            'out_subgoal': torch.stack(subgoals, dim=1),
            'out_progress': torch.stack(progresses, dim=1),
            'state_t_goal': state_t_goal,
            'state_t_instr': state_t_instr,
            'out_contrastive_data': contrastive_data,
        }
        return results

    def store_and_get_graph_feature(self, t_store_state, env_index, state_t_goal, state_t_instr):
        self.semantic_graph_implement.store_data_to_graph(
            store_state=t_store_state,
            env_index=env_index
        )
        global_graph_importent_features, global_graph_dict_objectIds_to_score = \
            self.semantic_graph_implement.chose_importent_node_feature(
                chose_type="GLOBAL_GRAPH",
                env_index=env_index,
                hidden_state=state_t_goal[0][env_index:env_index+1],
                )
        current_state_graph_importent_features, current_state_dict_objectIds_to_score = \
            self.semantic_graph_implement.get_graph_feature(
                chose_type="CURRENT_STATE_GRAPH",
                env_index=env_index,
                )
        history_changed_nodes_graph_importent_features, history_changed_dict_objectIds_to_score = \
            self.semantic_graph_implement.get_graph_feature(
                chose_type="HISTORY_CHANGED_NODES_GRAPH",
                env_index=env_index,
                )
        priori_importent_features, priori_dict_dict_objectIds_to_score = \
            self.semantic_graph_implement.chose_importent_node_feature(
                chose_type="PRIORI_GRAPH",
                env_index=env_index,
                hidden_state=state_t_goal[0][env_index:env_index+1],
                # hidden_state=state_t_instr[0][env_index:env_index+1],
                )
        return global_graph_importent_features, current_state_graph_importent_features,\
               history_changed_nodes_graph_importent_features, priori_importent_features,\
               global_graph_dict_objectIds_to_score, current_state_dict_objectIds_to_score,\
               history_changed_dict_objectIds_to_score, priori_dict_dict_objectIds_to_score

    # for model predict use Detailed Operation model || Navi model result
    def chose_embed_index(self, action_navi_t, action_oper_t, action_navi_or_operation_t):
        '''
        action_navi_t: embed index, [0~x]
        '''
        navi_or_operation_t = action_navi_or_operation_t.max(1)[1]
        w_t_navi = action_navi_t.max(1)[1]
        w_t_oper = action_oper_t.max(1)[1]
        w_t = [w_t_oper[i] if is_o else w_t_navi[i] for i, is_o in enumerate(navi_or_operation_t)]
        w_t = torch.stack(w_t)
        return w_t