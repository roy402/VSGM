import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import json
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
from eval_task import EvalTask
from env.thor_env import ThorEnv
import torch.multiprocessing as mp

import torch
import constants
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from icecream import ic
from models.utils.eval_debug import EvalDebug
eval_debug = EvalDebug()

classes = ['0'] + constants.OBJECTS + ['AppleSliced', 'ShowerCurtain', 'TomatoSliced', 'LettuceSliced', 'Lamp', 'ShowerHead', 'EggCracked', 'BreadSliced', 'PotatoSliced', 'Faucet']


class Leaderboard(EvalTask):
    '''
    dump action-sequences for leaderboard eval
    '''

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, splits, seen_actseqs, unseen_actseqs):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        while True:
            if task_queue.qsize() == 0:
                break

            task = task_queue.get()

            try:
                traj = model.load_task_json(task)
                r_idx = task['repeat_idx']
                print("Evaluating: %s" % (traj['root']))
                print("No. of trajectories left: %d" % (task_queue.qsize()))
                cls.evaluate(env, model, r_idx, resnet, traj, args, lock, splits, seen_actseqs, unseen_actseqs)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))

        # stop THOR
        env.stop()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, splits, seen_actseqs, unseen_actseqs):
        # reset model
        model.reset()

        # setup scene
        cls.setup_scene(env, traj_data, r_idx, args)

        # extract language features
        feat = model.featurize([traj_data], load_mask=False)

        # goal instr
        goal_instr = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        step_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs']
        current_high_descs = 0

        maskrcnn = maskrcnn_resnet50_fpn(num_classes=119)
        maskrcnn.eval()
        maskrcnn.load_state_dict(torch.load('weight_maskrcnn.pt'))
        maskrcnn = maskrcnn.cuda()

        prev_image = None
        prev_action = None
        nav_actions = ['MoveAhead_25', 'RotateLeft_90', 'RotateRight_90', 'LookDown_15', 'LookUp_15']
        
        prev_class = 0
        pred_class = 0
        prev_center = torch.zeros(2)

        done, success = False, False
        err = ""
        actions = list()
        fails = 0
        t = 0
        while not done:
            # break if max_steps reached
            if t >= args.max_steps:
                break

            # extract visual features
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
            curr_depth_image = env.last_event.depth_frame * (255 / 10000)
            curr_depth_image = curr_depth_image.astype(np.uint8)
            curr_instance = Image.fromarray(np.uint8(env.last_event.instance_segmentation_frame))
            feat = cls.get_frame_feat(cls, env, resnet, feat)
            feat['frames'] = feat['frames_conv']
            feat['all_meta_datas'] = cls.get_meta_datas(cls, env, resnet)
            # forward model
            m_out = model.step(feat)
            m_pred = model.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
            m_pred = list(m_pred.values())[0]

            # action prediction
            action = m_pred['action_low']
            if prev_image == curr_image and prev_action == action and prev_action in nav_actions and action in nav_actions and action == 'MoveAhead_25':
                dist_action = m_out['out_action_low'][0][0].detach().cpu()
                try:
                    idx_rotateR = model.vocab['action_low'].word2index('RotateRight_90')
                    idx_rotateL = model.vocab['action_low'].word2index('RotateLeft_90')
                except Exception as e:
                    idx_rotateR = model.action_low_word_to_index['RotateRight_90']
                    idx_rotateL = model.action_low_word_to_index['RotateLeft_90']
                action = 'RotateLeft_90' if dist_action[idx_rotateL] > dist_action[idx_rotateR] else 'RotateRight_90'

            # mask prediction
            mask = None
            if model.has_interaction(action):
                class_dist = m_pred['action_low_mask_label'][0]
                pred_class = np.argmax(class_dist)

                # mask generation
                with torch.no_grad():
                    out = maskrcnn([to_tensor(curr_image).cuda()])[0]
                    for k in out:
                        out[k] = out[k].detach().cpu()

                if sum(out['labels'] == pred_class) == 0:
                    mask = np.zeros((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
                else:
                    masks = out['masks'][out['labels'] == pred_class].detach().cpu()
                    scores = out['scores'][out['labels'] == pred_class].detach().cpu()

                    # Instance selection based on the minimum distance between the prev. and cur. instance of a same class.
                    if prev_class != pred_class:
                        scores, indices = scores.sort(descending=True)
                        masks = masks[indices]
                        prev_class = pred_class
                        prev_center = masks[0].squeeze(dim=0).nonzero().double().mean(dim=0)
                    else:
                        cur_centers = torch.stack([m.nonzero().double().mean(dim=0) for m in masks.squeeze(dim=1)])
                        distances = ((cur_centers - prev_center)**2).sum(dim=1)
                        distances, indices = distances.sort()
                        masks = masks[indices]
                        prev_center = cur_centers[0]

                    mask = np.squeeze(masks[0].numpy(), axis=0)

            '''
            eval_debug.add_data
            '''
            try:
                dict_action = {
                    # action
                    'action_low': m_pred["action_low"],
                    'action_navi_low': m_pred["action_navi_low"],
                    'action_operation_low': m_pred["action_operation_low"],
                    'action_navi_or_operation': m_pred["action_navi_or_operation"],
                    # goal
                    'subgoal_t': m_out["out_subgoal_t"],
                    'progress_t': m_out["out_progress_t"],
                    # ANALYZE_GRAPH
                    'global_graph_dict_ANALYZE_GRAPH': m_out["global_graph_dict_ANALYZE_GRAPH"],
                    'current_state_dict_ANALYZE_GRAPH': m_out["current_state_dict_ANALYZE_GRAPH"],
                    'history_changed_dict_ANALYZE_GRAPH': m_out["history_changed_dict_ANALYZE_GRAPH"],
                    'priori_dict_ANALYZE_GRAPH': m_out["priori_dict_ANALYZE_GRAPH"],
                    # mask
                    "mask": mask,
                    "pred_class": pred_class,
                    "object": classes[pred_class]
                }
            except Exception as e:
                ic(pred_class)
                ic(len(classes))

            eval_debug.add_data(t, curr_image, curr_depth_image, dict_action, "step_instr[current_high_descs]", err)

            if action == cls.STOP_TOKEN:
                print("\tpredicted STOP")
                break


            # use predicted action and mask (if available) to interact with the env
            t_success, _, _, err, api_action = env.va_interact(action, interact_mask=mask, smooth_nav=False)

            if not t_success:
                fails += 1
                if fails >= args.max_fails:
                    print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                    break

            # save action
            if api_action is not None:
                actions.append(api_action)

            # next time-step
            t += 1

            prev_image = curr_image
            prev_action = action
            pred_class = 0

        # check if goal was satisfied
        # eval_debug.record(model.args.dout, traj_data, goal_instr, step_instr, err, success)

        # actseq
        seen_ids = [t['task'] for t in splits['tests_seen']]
        actseq = {traj_data['task_id']: actions}

        # log action sequences
        lock.acquire()

        if traj_data['task_id'] in seen_ids:
            seen_actseqs.append(actseq)
        else:
            unseen_actseqs.append(actseq)

        lock.release()

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        seen_files, unseen_files = self.splits['tests_seen'], self.splits['tests_unseen']

        # add seen trajectories to queue
        for traj in seen_files:
            task_queue.put(traj)

        # add unseen trajectories to queue
        for traj in unseen_files:
            task_queue.put(traj)

        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()
        self.model.test_mode = True

        self.run(self.model, self.resnet, task_queue, self.args, lock, self.splits, self.seen_actseqs, self.unseen_actseqs)
        # for n in range(self.args.num_threads):
        #     thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
        #                                                self.splits, self.seen_actseqs, self.unseen_actseqs))
        #     thread.start()
        #     threads.append(thread)

        # for t in threads:
        #     t.join()

        # save
        self.save_results()

    def create_stats(self):
        '''
        storage for seen and unseen actseqs
        '''
        self.seen_actseqs, self.unseen_actseqs = self.manager.list(), self.manager.list()

    def save_results(self):
        '''
        save actseqs as JSONs
        '''
        results = {'tests_seen': list(self.seen_actseqs),
                   'tests_unseen': list(self.unseen_actseqs)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'tests_actseqs_dump_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)


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

    '''
    Semantic graph map
    '''
    import os
    import sys
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
    sys.path.append(os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))

    parser.add_argument("config_file", default="models/config/without_env_base.yaml", help="path to config file")
    parser.add_argument("--semantic_config_file", default="models/config/mini_moca_graph_softmaxgcn.yaml", help="path to config file")
    parser.add_argument("--sgg_config_file", default=None, help="path to config file $GRAPH_RCNN_ROOT/configs/attribute.yaml")
    parser.add_argument('--gpu_id', help='use gpu 0/1', default=1, type=int)
    parser.add_argument("-p", "--params", nargs="+", metavar="my.setting=value", default=[],
                        help="override params of the config file,"
                             " e.g. -p 'training.gamma=0.95'")
    '''
    Semantic graph map
    '''
    # settings
    parser.add_argument('--splits', type=str, default="data/splits/oct21.json")
    parser.add_argument('--data', type=str, default="data/json_feat_2.1.0")
    parser.add_argument('--model_path', type=str, default="exp/pretrained/pretrained.pth")
    parser.add_argument('--model', type=str, default='models.model.seq2seq_im_mask')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true')
    parser.add_argument('--gpu', dest='gpu', action='store_true', default=True)
    parser.add_argument('--num_threads', type=int, default=4)


    args = parser.parse_args()

    config = load_config(args)
    args.config_file = config


    # parse arguments

    # fixed settings (DO NOT CHANGE)
    args.max_steps = 1000
    args.max_fails = 10

    # leaderboard dump
    eval = Leaderboard(args, manager)

    # start threads
    eval.spawn_threads()
