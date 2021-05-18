import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
sys.path.append(os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
import argparse
import torch.multiprocessing as mp
from eval_task import EvalTask
from eval_subgoals import EvalSubgoals


def load_config(args):
    import yaml
    import glob
    assert os.path.exists(args.config_file), "Invalid config file "
    with open(args.config_file) as reader:
        config = yaml.safe_load(reader)
    # Parse overriden params.
    for param in args.params:
        fqn_key, value = param.split("=")
        entry_to_change = config
        keys = fqn_key.split(".")
        for k in keys[:-1]:
            entry_to_change = entry_to_change[k]
        entry_to_change[keys[-1]] = yaml.load(value)

    ### other ###
    if args.semantic_config_file is not None:
        sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
        from config import cfg
        cfg.merge_from_file(args.semantic_config_file)
        cfg.GENERAL.save_path = cfg.GENERAL.save_path + sys.argv[0].split("/")[-1] + "_"
        config['semantic_cfg'] = cfg
        config["general"]["save_path"] = cfg.GENERAL.save_path
        config["vision_dagger"]["use_exploration_frame_feats"] = cfg.GENERAL.use_exploration_frame_feats
    if args.sgg_config_file is not None:
        sys.path.insert(0, os.environ['GRAPH_RCNN_ROOT'])
        from lib.config import cfg
        cfg.merge_from_file(args.sgg_config_file)
        config['sgg_cfg'] = cfg
    # print(config)

    return config


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", default="models/config/without_env_base.yaml", help="path to config file")
    parser.add_argument("--semantic_config_file", default="models/config/mini_moca_graph_softmaxgcn.yaml", help="path to config file")
    parser.add_argument("--sgg_config_file", default=None, help="path to config file")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")

    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_2.1.0")
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--eval_split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])
    parser.add_argument('--model_path', type=str, default="model.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true')
    parser.add_argument('--gpu_id', help='use gpu 0/1', default=1, type=int)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--gcn_cat_visaul', help='use visual embedding to gcn', action='store_true')

    # eval params
    parser.add_argument('--max_steps', type=int, default=1000, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')

    # eval settings
    parser.add_argument('--subgoals', type=str, help="subgoals to evaluate independently, eg:all or GotoLocation,PickupObject...", default="")
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--skip_model_unroll_with_expert', action='store_true', help='forward model with expert actions')
    parser.add_argument('--no_teacher_force_unroll_with_expert', action='store_true', help='no teacher forcing with expert')

    # debug
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--fast_epoch', dest='fast_epoch', action='store_true')

    parser.add_argument('--task_types', type=str, help="task_types", default="1,2,3,4,5,6")
    # parse arguments
    args = parser.parse_args()
    config = load_config(args)
    args.config_file = config
    # import torch
    # device = torch.device("cuda:%d" % args.gpu_id if args.gpu else "cpu")
    # if args.gpu and torch.cuda.is_available():
    #     torch.cuda.set_device(device)
    # eval mode
    if args.subgoals:
        eval = EvalSubgoals(args, manager)
    else:
        eval = EvalTask(args, manager)

    # start threads
    eval.spawn_threads()