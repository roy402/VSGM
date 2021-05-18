import datetime
import os
import random
import time
import copy
import json
import glob
import importlib
import numpy as np

import sys
sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))

from agent import OracleSggDAggerAgent
import modules.generic as generic
import torch
from eval.evaluate_semantic_graph_dagger import evaluate_semantic_graph_dagger
from modules.generic import HistoryScoreCache, EpisodicCountingMemory, ObjCentricEpisodicMemory
from agents.utils.misc import extract_admissible_commands
from agents.utils.traj_process import get_traj_train_data, get_exploration_traj_train_data
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb

def train():

    time_1 = datetime.datetime.now()
    config = generic.load_config()
    # pdb.set_trace()
    agent = OracleSggDAggerAgent(config)
    env_type = "AlfredThorEnv"
    alfred_env = getattr(importlib.import_module("environment"), env_type)(config, train_eval="train")
    # env = alfred_env.init_env(batch_size=1)
    json_file_list = alfred_env.json_file_list
    # print(json_file_list)
    # import pdb;pdb.set_trace()

    id_eval_env, num_id_eval_game = None, 0
    ood_eval_env, num_ood_eval_game = None, 0
    if agent.run_eval:
        config['env']['thor']['save_frames_to_disk'] = config['semantic_cfg'].GENERAL.SAVE_EVAL_FRAME
        # in distribution
        if config['dataset']['eval_id_data_path'] is not None:
            alfred_env = getattr(importlib.import_module("environment"), env_type)(config, train_eval="eval_in_distribution")
            id_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_id_eval_game = alfred_env.num_games
        # out of distribution
        if config['dataset']['eval_ood_data_path'] is not None:
            alfred_env = getattr(importlib.import_module("environment"), env_type)(config, train_eval="eval_out_of_distribution")
            ood_eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)
            num_ood_eval_game = alfred_env.num_games

    output_dir = config["general"]["save_path"]
    data_dir = config["general"]["save_path"]
    action_space = config["dagger"]["action_space"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    step_in_total = 0
    episode_no = 0
    running_avg_student_points = HistoryScoreCache(capacity=500)
    running_avg_student_steps = HistoryScoreCache(capacity=500)
    running_avg_dagger_loss = HistoryScoreCache(capacity=500)

    json_file_name = agent.experiment_tag.replace(" ", "_")
    best_performance_so_far = 0.0
    best_performance_loss = 1000

    # load model from checkpoint
    if agent.load_pretrained:
        print(data_dir + "/" + agent.load_from_tag + ".pt")
        print(os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"))
        if os.path.exists(data_dir + "/" + agent.load_from_tag + ".pt"):
            print("load model")
            agent.load_pretrained_model(data_dir + "/" + agent.load_from_tag + ".pt")
            agent.update_target_net()

    while(True):
        if episode_no > agent.max_episode:
            break
        np.random.seed(episode_no)
        print("reload")
        print("reload ends")
        batch_size = agent.batch_size
        tasks = random.sample(json_file_list, k=batch_size)
        # print("tasks: ", tasks)
        save_frames_path = config['dataset']['presave_data_path']
        transition_caches = get_traj_train_data(tasks, save_frames_path)
        if agent.use_exploration_frame_feats:
            exploration_transition_caches = get_exploration_traj_train_data(
                tasks,
                save_frames_path
            )

        report = agent.report_frequency > 0 and (episode_no % agent.report_frequency <= (episode_no - batch_size) % agent.report_frequency)
        agent.train()
        agent.init(batch_size)

        # extract exploration frame features
        '''
        # Add exploration img & meta data to GraphData
        '''
        for loop in range(10):
            if agent.use_exploration_frame_feats:
                for env_index in range(len(exploration_transition_caches)):
                    exploration_transition_cache = exploration_transition_caches[env_index]
                    store_states = exploration_transition_cache[0]
                    agent.update_exploration_data_to_global_graph(store_states, env_index)
            losses = []
            for env_index in range(len(transition_caches)):
                transition_cache = transition_caches[env_index]
                store_states, task_desc_strings, expert_actions = transition_cache[0], transition_cache[1], transition_cache[2]
                head = np.random.randint(len(store_states))
                loss = agent.train_command_generation_recurrent_teacher_force(
                    store_states[head:head + agent.dagger_replay_sample_history_length],
                    task_desc_strings[head:head + agent.dagger_replay_sample_history_length],
                    expert_actions[head:head + agent.dagger_replay_sample_history_length],
                    train_now=False,
                    env_index=env_index,
                )
                loss_copy = loss.clone().detach()
                losses.append(loss)
                running_avg_dagger_loss.push(loss_copy.item())
            loss = torch.stack(losses).mean()
            loss = agent.grad(loss)
            print("loss: ", loss.item())
            agent.summary_writer.training_loss(
                train_loss=loss,
                optimizer=agent.optimizer
            )
            agent.reset_all_scene_graph()

        agent.finish_of_episode(episode_no, batch_size, decay_lr=True)
        episode_no += batch_size
        print("episode_no: ", episode_no)

        print("running_avg_dagger_loss.get_avg: ", running_avg_dagger_loss.get_avg())
        print("best_performance_so_far: ", best_performance_so_far)
        print("best_performance_loss: ", best_performance_loss)
        if not report:
            continue
        time_2 = datetime.datetime.now()
        time_spent_seconds = (time_2-time_1).seconds
        eps_per_sec = float(episode_no) / time_spent_seconds
        # evaluate
        print("Save Model")

        id_eval_game_points, id_eval_game_step, id_eval_game_goal_condition_points = 0.0, 0.0, 0.0
        ood_eval_game_points, ood_eval_game_step, ood_eval_game_goal_condition_points = 0.0, 0.0, 0.0
        if agent.run_eval:
            if id_eval_env is not None:# and episode_no != batch_size:
                id_eval_res, _ = evaluate_semantic_graph_dagger(id_eval_env, agent, num_id_eval_game)
                id_eval_game_points, id_eval_game_step = id_eval_res['average_points'], id_eval_res['average_steps']
                id_eval_game_goal_condition_points = id_eval_res['average_goal_condition_points']
            if ood_eval_env is not None:# and episode_no != batch_size:
                ood_eval_res, _ = evaluate_semantic_graph_dagger(ood_eval_env, agent, num_ood_eval_game)
                ood_eval_game_points, ood_eval_game_step = ood_eval_res['average_points'], ood_eval_res['average_steps']
                ood_eval_game_goal_condition_points = ood_eval_res['average_goal_condition_points']
            if id_eval_game_points >= best_performance_so_far:
                best_performance_so_far = id_eval_game_points
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")
            agent.summary_writer.eval(
                id_eval_game_points,
                id_eval_game_step,
                id_eval_game_goal_condition_points,
                ood_eval_game_points,
                ood_eval_game_step,
                ood_eval_game_goal_condition_points
            )
        else:
            if running_avg_dagger_loss.get_avg() <= best_performance_loss:
                best_performance_loss = running_avg_dagger_loss.get_avg()
                agent.save_model_to_path(output_dir + "/" + agent.experiment_tag + ".pt")
        print("Save Model end")
        print("dagger_replay_sample_history_length: ", agent.dagger_replay_sample_history_length)
        agent.dagger_replay_sample_history_length += 1

        # write accuracies down into file
        _s = json.dumps({"time spent": str(time_2 - time_1).rsplit(".")[0],
                         "time spent seconds": time_spent_seconds,
                         "episodes": episode_no,
                         "episodes per second": eps_per_sec,
                         "loss": str(running_avg_dagger_loss.get_avg())})
        with open(output_dir + "/" + json_file_name + '.json', 'a+') as outfile:
            outfile.write(_s + '\n')
            outfile.flush()


if __name__ == '__main__':
    train()
