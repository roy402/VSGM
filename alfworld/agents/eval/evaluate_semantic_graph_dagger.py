import copy
import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
from agents.utils.misc import extract_admissible_commands


def evaluate_semantic_graph_dagger(env, agent, num_games, debug=False):
    env.seed(42)
    agent.eval()
    episode_no = 0
    res_points, res_steps, res_gcs = [], [], []
    res_info = []
    final_dynamics = {}
    with torch.no_grad():
        while(True):
            print("=== eval env.index_save_train_data === ", env.index_save_train_data)
            print("=== eval env.json_file_list === ", len(env.json_file_list))
            if episode_no >= num_games:
                break
            obs, infos = env.reset()
            game_names = infos["extra.gamefile"]
            batch_size = len(obs)
            if agent.unstick_by_beam_search:
                smart = [{"not working": [], "to try": []} for _ in range(batch_size)]

            agent.init(batch_size)
            previous_dynamics = None

            execute_actions = []
            prev_step_dones, prev_rewards = [], []
            for _ in range(batch_size):
                execute_actions.append("restart")
                prev_step_dones.append(0.0)
                prev_rewards.append(0.0)

            if agent.action_space == "exhaustive":
                raise NotImplementedError()
            else:
                action_candidate_list = list(infos["admissible_commands"])
            action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)

            # extract exploration frame features
            if agent.use_exploration_frame_feats:
                exploration_frames = env.get_exploration_frames()
                exploration_sgg_meta_datas = env.get_sgg_meta_datas()
                for env_index in range(len(exploration_frames)):
                    exploration_frame = exploration_frames[env_index]
                    exploration_sgg_meta_data = exploration_sgg_meta_datas[env_index]
                    store_states = []
                    for one_episode_frames, one_episode_sgg_meta_data in zip(exploration_frame, exploration_sgg_meta_data):
                        store_state = {
                            "exploration_img": one_episode_frames,
                            "exploration_sgg_meta_data": one_episode_sgg_meta_data
                        }
                        store_states.append(store_state)
                    agent.update_exploration_data_to_global_graph(store_states, env_index)

            observation_strings = list(obs)
            observation_strings = agent.preprocess_observation(observation_strings)
            task_desc_strings, observation_strings = agent.get_task_and_obs(observation_strings)

            still_running_mask = []
            sequence_game_points = []
            goal_condition_points = []
            print_actions = []
            report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= (episode_no - batch_size) % agent.report_frequency)

            for step_no in range(agent.max_nb_steps_per_episode):
                # get visual features
                observation_feats, dict_objectIds_to_scores = None, []
                for env_index in range(len(env.envs)):
                    if previous_dynamics is not None:
                        hidden_state = previous_dynamics[env_index].unsqueeze(0)
                    else:
                        hidden_state = None
                    observation_feat, store_state, dict_objectIds_to_score = agent.extract_visual_features(
                        thor=env.envs[env_index],
                        hidden_state=hidden_state,
                        env_index=env_index
                        )
                    if observation_feats is None:
                        observation_feats = observation_feat
                    else:
                        observation_feats.extend(observation_feat)
                    dict_objectIds_to_scores.append(dict_objectIds_to_score)

                # predict actions
                if agent.action_space == "generation":
                    prev_actions = copy.copy(execute_actions)
                    if agent.unstick_by_beam_search:
                        for i in range(batch_size):
                            if "Nothing happens" in observation_strings[i]:
                                smart[i]["not working"].append(execute_actions[i])

                    execute_actions, current_dynamics = agent.command_generation_greedy_generation(
                        observation_feats,
                        task_desc_strings,
                        previous_dynamics,
                    )
                    if agent.unstick_by_beam_search:
                        for i in range(batch_size):
                            if "Nothing happens" in observation_strings[i] and execute_actions[i] in smart[i]["not working"]:
                                if len(smart[i]["to try"]) == 0:
                                    bs_actions, _ = agent.command_generation_beam_search_generation(
                                        observation_feats[i: i + 1],
                                        task_desc_strings[i: i + 1],
                                        None if previous_dynamics is None else previous_dynamics[i: i + 1]
                                    )
                                    bs_actions = bs_actions[0]
                                    smart[i]["to try"] += bs_actions

                                smart[i]["to try"] = [item for item in smart[i]["to try"] if item != prev_actions[i]]
                                if len(smart[i]["to try"]) > 0:
                                    execute_actions[i] = smart[i]["to try"][0]
                            else:
                                smart[i] = {"not working": [], "to try": []}  # reset
                else:
                    raise NotImplementedError()

                obs, _, dones, infos = env.step(execute_actions)
                scores = [float(item) for item in infos["won"]]
                gcs = [float(item) for item in infos["goal_condition_success_rate"]] if "goal_condition_success_rate" in infos else [0.0]*batch_size
                dones = [float(item) for item in dones]

                if debug:
                    print(execute_actions[0])
                    print(obs[0])

                if agent.action_space == "exhaustive":
                    raise NotImplementedError()
                else:
                    action_candidate_list = list(infos["admissible_commands"])
                action_candidate_list = agent.preprocess_action_candidates(action_candidate_list)
                previous_dynamics = current_dynamics
                if agent.ANALYZE_GRAPH:
                    expert_actions = []
                    for b in range(batch_size):
                        next_action = "None"
                        if "expert_plan" in infos and len(infos["expert_plan"][b]) > 0:
                            next_action = infos["expert_plan"][b][0]
                        expert_actions.append(next_action)
                    env.store_analyze_graph(dict_objectIds_to_scores, expert_actions, execute_actions)

                if step_no == agent.max_nb_steps_per_episode - 1:
                    # terminate the game because DQN requires one extra step
                    dones = [1.0 for _ in dones]

                still_running = [1.0 - float(item) for item in prev_step_dones]  # list of float
                prev_step_dones = dones
                step_rewards = [float(curr) - float(prev) for curr, prev in zip(scores, prev_rewards)]  # list of float
                prev_rewards = scores
                sequence_game_points.append(step_rewards)
                goal_condition_points.append(gcs)
                still_running_mask.append(still_running)
                print_actions.append(execute_actions[0] if still_running[0] else "--")

                # if all ended, break
                if np.sum(still_running) == 0:
                    break

            for n in range(batch_size):
                Thor = env.envs[n]
                task_name = Thor.env.save_frames_path
                final_dynamics[task_name] = {}
                final_dynamics[task_name]["final_dynamics"] = current_dynamics[n].to('cpu').numpy()
                final_dynamics[task_name]["observation_feats"] = observation_feats[n].to('cpu').numpy()
                final_dynamics[task_name]["label"] = Thor.traj_data['task_type']

            game_steps = np.sum(np.array(still_running_mask), 0).tolist()  # batch
            game_points = np.max(np.array(sequence_game_points), 0).tolist()  # batch
            game_gcs = np.max(np.array(goal_condition_points), 0).tolist() # batch
            for i in range(batch_size):
                if len(res_points) >= num_games:
                    break
                res_points.append(game_points[i])
                res_gcs.append(game_gcs[i])
                res_steps.append(game_steps[i])
                res_info.append("/".join(game_names[i].split("/")[-3:-1]) + ", score: " + str(game_points[i]) + ", step: " + str(game_steps[i]))

            # finish game
            if agent.ANALYZE_GRAPH:
                env.one_episode_analyze_graph_end(game_points, game_gcs)
            agent.finish_of_episode(episode_no, batch_size, decay_lr=False)
            episode_no += batch_size

            if not report:
                continue
            print("Model: {:s} | Episode: {:3d} | {:s} |  game points: {:2.3f} | game goal-condition points: {:2.3f} | game steps: {:2.3f}".format(agent.experiment_tag, episode_no, game_names[0], np.mean(res_points), np.mean(res_gcs), np.mean(res_steps)))
            # print(game_id + ":    " + " | ".join(print_actions))
            print(" | ".join(print_actions))


        average_points, average_gc_points, average_steps = np.mean(res_points), np.mean(res_gcs), np.mean(res_steps)
        print("================================================")
        print("eval game points: " + str(average_points) + ", eval game goal-condition points : " + str(average_gc_points) + ", eval game steps: " + str(average_steps))
        for item in res_info:
            print(item)

        return {
            'average_points': average_points,
            'average_goal_condition_points': average_gc_points,
            'average_steps': average_steps,
            'res_points': res_points,
            'res_gcs': res_gcs,
            'res_steps': res_steps,
            'res_info': res_info
        }, final_dynamics
