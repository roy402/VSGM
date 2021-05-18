import copy
import operator
import logging
from queue import PriorityQueue
import numpy as np
import torch
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
from agent import BaseAgent
import modules.memory as memory
from modules.generic import to_np, to_pt, _words_to_ids, pad_sequences, preproc, max_len, ez_gather_dim_1, LinearSchedule, BeamSearchNode
from modules.layers import NegativeLogLoss, masked_mean, compute_mask, GetGenerationQValue
from agent import OracleSggDAggerAgent


class ButlerSemanticAgent(OracleSggDAggerAgent):
    def __init__(self, config):
        super().__init__(config)

    def update_dagger(self):
        if self.recurrent:
            return self.train_dagger_recurrent()
        else:
            return self.train_dagger()

    def train_command_generation_recurrent_teacher_force(self, observation, seq_task_desc_strings, seq_target_strings, contains_first_step=False, train_now=True, env_index=None):
        seq_observation_strings = [o["most_recent_observation_strings"] for o in observation]
        store_states = [o["store_states"] for o in observation]

        loss_list = []
        previous_dynamics = None
        batch_size = len(seq_target_strings[0])
        # task_desc_strings
        h_td, td_mask = self.encode(seq_task_desc_strings[0], use_model="online")
        for step_no in range(len(seq_target_strings)):
            input_target_strings = [" ".join(["[CLS]"] + item.split()) for item in seq_target_strings[step_no]]
            output_target_strings = [" ".join(item.split() + ["[SEP]"]) for item in seq_target_strings[step_no]]
            # observation_strings
            input_obs = self.get_word_input(seq_observation_strings[step_no])
            h_obs, obs_mask = self.encode([seq_observation_strings[step_no]], use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(
                h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
            # vision feat
            observation_feats, _, _ = self.extract_visual_features(
                store_state=store_states[step_no],
                hidden_state=previous_dynamics,
                env_index=env_index)
            h_vision_obs = self._vision_feat(observation_feats, h_obs.device)
            aggregated_obs_representation = torch.cat((h_vision_obs, aggregated_obs_representation), dim=1)
            obs_mask = torch.ones((batch_size, h_vision_obs.shape[1]+obs_mask.shape[1])).to(obs_mask.device)

            # current_dynamics
            averaged_representation = self.online_net.masked_mean(
                aggregated_obs_representation, obs_mask)  # batch x hid
            current_dynamics = self.online_net.rnncell(
                averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)

            input_target = self.get_word_input(input_target_strings)
            ground_truth = self.get_word_input(output_target_strings)  # batch x target_length
            target_mask = compute_mask(input_target)  # mask of ground truth should be the same
            pred = self.online_net.vision_decode(
                input_target,
                target_mask,
                aggregated_obs_representation,
                obs_mask,
                current_dynamics)  # batch x target_length x vocab

            previous_dynamics = current_dynamics
            if (not contains_first_step) and step_no < self.dagger_replay_sample_update_from:
                previous_dynamics = previous_dynamics.detach()
                continue

            batch_loss = NegativeLogLoss(pred * target_mask.unsqueeze(-1), ground_truth, target_mask, smoothing_eps=self.smoothing_eps)
            loss = torch.mean(batch_loss)
            loss_list.append(loss)
        if len(loss_list) == 0:
            return None
        loss = torch.stack(loss_list).mean()
        # print("loss: ", loss)
        if train_now:
            loss = self.grad(loss)
            return loss
        else:
            return loss

    def _vision_feat(self, observation_feats, device):
        obs = [o.to(device) for o in observation_feats]
        # torch.Size([1, 1024])
        aggregated_obs_feat = self.aggregate_feats_seq(obs)
        # torch.Size([1, 1, 64])
        h_vision_obs = self.online_net.vision_fc(aggregated_obs_feat)
        return h_vision_obs

    def command_generation_greedy_generation(self, observation, task_desc_strings, previous_dynamics):
        with torch.no_grad():
            observation_strings = observation["most_recent_observation_strings"]
            observation_feats = observation["observation_feats"]

            batch_size = len(observation_strings)
            # observation_strings
            input_obs = self.get_word_input(observation_strings)
            # torch.Size([2, 101, 64]), torch.Size([2, 101])
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            # task_desc_strings
            # torch.Size([2, 12, 64]), torch.Size([2, 12])
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(
                h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid
            # vision feat
            h_vision_obs = self._vision_feat(observation_feats, h_obs.device)
            aggregated_obs_representation = torch.cat((h_vision_obs, aggregated_obs_representation), dim=1)
            obs_mask = torch.ones((batch_size, h_vision_obs.shape[1]+obs_mask.shape[1])).to(obs_mask.device)

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(
                    aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
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
                pred = self.online_net.vision_decode(
                    input_target,
                    target_mask,
                    aggregated_obs_representation,
                    obs_mask,
                    current_dynamics)  # batch x target_length x vocab
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

    def command_generation_beam_search_generation(self, observation_strings, task_desc_strings, previous_dynamics):
        with torch.no_grad():

            batch_size = len(observation_strings)
            beam_width = self.beam_width
            if beam_width == 1:
                res, current_dynamics = self.command_generation_greedy_generation(observation_strings, task_desc_strings, previous_dynamics)
                res = [[item] for item in res]
                return res, current_dynamics
            generate_top_k = self.generate_top_k
            res = []

            input_obs = self.get_word_input(observation_strings)
            h_obs, obs_mask = self.encode(observation_strings, use_model="online")
            h_td, td_mask = self.encode(task_desc_strings, use_model="online")
            aggregated_obs_representation = self.online_net.aggretate_information(h_obs, obs_mask, h_td, td_mask)  # batch x obs_length x hid

            if self.recurrent:
                averaged_representation = self.online_net.masked_mean(aggregated_obs_representation, obs_mask)  # batch x hid
                current_dynamics = self.online_net.rnncell(averaged_representation, previous_dynamics) if previous_dynamics is not None else self.online_net.rnncell(averaged_representation)
            else:
                current_dynamics = None

            for b in range(batch_size):

                # starts from CLS tokens
                __input_target_list = [self.word2id["[CLS]"]]
                __input_obs = input_obs[b: b + 1]  # 1 x obs_len
                __obs_mask = obs_mask[b: b + 1]  # 1 x obs_len
                __aggregated_obs_representation = aggregated_obs_representation[b: b + 1]  # 1 x obs_len x hid
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
                    pred = self.online_net.vision_decode(
                        input_target,
                        target_mask,
                        __aggregated_obs_representation,
                        __obs_mask,
                        __current_dynamics,)  # 1 x target_length x vocab
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
