import os
import json
import importlib

import sys
sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))

from agent import OracleSggDAggerAgent
from agents.semantic_graph.utils import save_final_dynamics
import modules.generic as generic
from eval.evaluate_semantic_graph_dagger import evaluate_semantic_graph_dagger
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_eval():
    config = generic.load_config()
    agent = OracleSggDAggerAgent(config)

    output_dir = "./runs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load model from checkpoint
    data_dir = config['semantic_cfg'].GENERAL.LOAD_PRETRAINED_PATH
    if agent.load_pretrained:
        if os.path.exists(data_dir + ".pt"):
            agent.load_pretrained_model(data_dir + ".pt")
            agent.update_target_net()

    training_method = config["general"]["training_method"]
    eval_paths = config["general"]["evaluate"]["eval_paths"]
    eval_envs = config["general"]["evaluate"]["envs"]
    controllers = config["general"]["evaluate"]["controllers"]
    repeats = config["general"]["evaluate"]["repeats"]

    # iterate through all environments
    for eval_env_type in eval_envs:
        # iterate through all controllers
        for controller_type in (controllers if eval_env_type == "AlfredThorEnv" else ["tw"]):
            print("Setting controller: %s" % controller_type)
            # iterate through all splits
            for eval_path in eval_paths:
                print("Evaluating: %s" % eval_path)
                config["general"]["evaluate"]["env"]["type"] = eval_env_type
                config["dataset"]["eval_ood_data_path"] = eval_path
                config["controller"]["type"] = controller_type

                alfred_env = getattr(importlib.import_module("environment"), "AlfredThorEnv")(config, train_eval="eval_out_of_distribution")
                eval_env = alfred_env.init_env(batch_size=agent.eval_batch_size)

                # evaluate method
                if training_method == "dagger":
                    results, final_dynamics = evaluate_semantic_graph_dagger(eval_env, agent, alfred_env.num_games*repeats)
                    final_dynamics_path = eval_env.config['env']['thor']['save_frames_path']
                    name = eval_path.split("/")[-1] + "_final_dynamics"
                    save_final_dynamics(final_dynamics_path, final_dynamics, name=name)
                elif training_method == "dqn":
                    results = evaluate_dqn(eval_env, agent, alfred_env.num_games*repeats)
                else:
                    raise NotImplementedError()

                # save results to json
                split_name = eval_path.split("/")[-1]
                experiment_name = config["general"]["evaluate"]["eval_experiment_tag"]
                results_json = os.path.join(output_dir, "{}_{}_{}_{}.json".format(experiment_name, eval_env_type.lower(), controller_type, split_name))

                with open(results_json, 'w') as f:
                    json.dump(results, f, indent=4, sort_keys=True)
                print("Saved %s" % results_json)

                eval_env.close()


if __name__ == '__main__':
    run_eval()
