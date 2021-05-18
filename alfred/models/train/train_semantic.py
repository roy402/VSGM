import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import os
import torch
import pprint
import json
import time
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to
# SET ALFRED_ROOT=D:\alfred


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

    output_dir = config["general"]["save_path"]
    if output_dir != '.' and args.semantic_config_file is not None and not args.not_save_config:
        from shutil import copyfile
        # import pdb; pdb.set_trace()
        folder_count = len(glob.glob(os.path.join(output_dir + "*", "")))
        if folder_count != 0:
            folder_count = glob.glob(os.path.join(output_dir + "*", ""))
            folder_count.sort()
            folder_count = folder_count[-1]
            folder_count = folder_count[len(output_dir):-1]
            folder_count = int(folder_count) + 1
        output_dir += str(folder_count)
        config["general"]["save_path"] = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        dst_file = os.path.join(output_dir, args.config_file.split('/')[-1])
        copyfile(args.config_file, dst_file)
        if args.semantic_config_file is not None:
            dst_file = os.path.join(output_dir, args.semantic_config_file.split('/')[-1])
            copyfile(args.semantic_config_file, dst_file)
        if args.sgg_config_file is not None:
            dst_file = os.path.join(output_dir, args.sgg_config_file.split('/')[-1])
            copyfile(args.sgg_config_file, dst_file)
    return config


if __name__ == '__main__':
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("config_file", default="models/config/without_env_base.yaml", help="path to config file")
    parser.add_argument("--semantic_config_file", default="models/config/mini_moca_graph_softmaxgcn.yaml", help="path to config file")
    parser.add_argument("--sgg_config_file", default=None, help="path to config file $GRAPH_RCNN_ROOT/configs/attribute.yaml")
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")
    parser.add_argument('--not_save_config', help='test without save config', action='store_true')

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder data/json_feat_2.1.0, data/full_2.1.0', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use seq2seq_im/gcn_im', default='seq2seq_im')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--gpu_id', help='use gpu 0/1', default="cuda", type=str)
    parser.add_argument('--dout', help='where to save model', default='exp/model,{model}')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=4, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=50, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=3*7*7, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0.2, type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0.2, type=float)
    parser.add_argument('--action_navi_loss_wt', help='weight of action loss', default=0.5, type=float)
    parser.add_argument('--action_oper_loss_wt', help='weight of action loss', default=0.5, type=float)
    parser.add_argument('--action_navi_or_oper_loss_wt', help='weight of action loss', default=0.5, type=float)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--mask_label_loss_wt', help='weight of mask loss', default=1., type=float)

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--fast_epoch', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)

    # graph model
    parser.add_argument('--model_hete_graph', help='', action='store_true')
    parser.add_argument('--HETAttention', help='', action='store_true')
    parser.add_argument('--HetLowSg', help='', action='store_true')
    parser.add_argument('--augmentation', help='', action='store_true')

    # contrastive learning
    parser.add_argument('--DataParallelDevice', action='append', type=int)
    parser.add_argument('--contrastive_loss_wt', help='weight of progress monitor', default=0.5, type=float)
    parser.add_argument('--contrastive_margin', help='weight of progress monitor', default=0.1, type=float)
    parser.add_argument('--print', help='', action='store_true')

    parser.add_argument('--task_types', type=str, help="task_types", default="1,2,3,4,5,6")
    # MOCAMaskDepthGraph_V5
    # 1024*18*18 -> 1024*9*9
    # self.conv_pool = torch.nn.MaxPool2d(sgg_pool, stride=sgg_pool)
    parser.add_argument('--sgg_pool', type=int, default=2)

    # args and init
    args = parser.parse_args()

    config = load_config(args)
    args.config_file = config

    args.dout = args.dout.format(**vars(args)) + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    torch.manual_seed(args.seed)

    # device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(device)

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    # load model
    M = import_module('model.{}'.format(args.model))
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer = M.Module.load(args.resume)
    else:
        model = M.Module(args, vocab)
        optimizer = None
    # to gpu
    torch.cuda.device_count()
    # torch.cuda.set_device('cuda:%d' % args.gpu_id)
    # print(torch.cuda.current_device())

    if args.gpu:
        from torch import nn
        model = model.to(args.gpu_id)
        if not optimizer is None:
            optimizer_to(optimizer, torch.device(args.gpu_id))

    # start train loop
    model.run_train(splits, optimizer=optimizer)