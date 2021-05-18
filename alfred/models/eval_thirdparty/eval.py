import json
import pprint
import random
import time
import torch
import torch.multiprocessing as mp
from models.nn.resnet import Resnet
from data.preprocess import Dataset
from importlib import import_module
import numpy as np
from PIL import Image
import os


class Eval(object):

    # tokens
    STOP_TOKEN = "<<stop>>"
    SEQ_TOKEN = "<<seg>>"
    TERMINAL_TOKENS = [STOP_TOKEN, SEQ_TOKEN]

    def __init__(self, args, manager):
        # args and manager
        self.args = args
        self.manager = manager

        # load splits
        with open(self.args.splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        # load model
        print("Loading: ", self.args.model_path)
        # M = import_module(self.args.model)
        M = import_module('model.{}'.format(args.model))
        self.model, optimizer = M.Module.load(self.args.model_path)
        self.model.share_memory()
        self.model.eval()
        self.model.test_mode = True

        # updated args
        self.model.args.dout = self.args.model_path.replace(self.args.model_path.split('/')[-1], '')
        self.model.args.data = self.args.data if self.args.data else self.model.args.data

        # preprocess and save
        if args.preprocess:
            print("\nPreprocessing dataset and saving to %s folders ... This is will take a while. Do this once as required:" % self.model.args.pp_folder)
            self.model.args.fast_epoch = self.args.fast_epoch
            dataset = Dataset(self.model.args, self.model.vocab)
            dataset.preprocess_splits(self.splits)

        # load resnet
        args.visual_model = 'resnet18'

        # gpu
        if self.args.gpu:
            self.model = self.model.to(torch.device('cuda'))
            self.resnet = Resnet(args, device='cuda', eval=True, share_memory=True, use_conv_feat=True)
        else:
            self.resnet = Resnet(args, device='cpu', eval=True, share_memory=True, use_conv_feat=True)
        if args.sgg_config_file is not None:
            import sys
            sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents', 'sgg'))
            from sgg.sgg import load_pretrained_model
            cfg_sgg = args.config_file['sgg_cfg']
            self.resnet = load_pretrained_model(
                cfg_sgg,
                self.model.semantic_graph_implement.trans_MetaData.transforms,
                self.model.semantic_graph_implement.trans_MetaData.SGG_result_ind_to_classes,
                'cuda'
                )
        # success and failure lists
        self.create_stats()

        # set random seed for shuffling
        random.seed(int(time.time()))

    def queue_tasks(self):
        '''
        create queue of trajectories to be evaluated
        '''
        task_queue = self.manager.Queue()
        files = self.splits[self.args.eval_split]

        # debugging: fast epoch
        if self.args.fast_epoch:
            files = files[:16]

        '''
        new code
        '''
        TASK_TYPES = {"1": "pick_and_place_simple",
                      "2": "look_at_obj_in_light",
                      "3": "pick_clean_then_place_in_recep",
                      "4": "pick_heat_then_place_in_recep",
                      "5": "pick_cool_then_place_in_recep",
                      "6": "pick_two_obj_and_place"}
        task_types = []
        for tt_id in self.args.task_types.split(','):
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        if self.args.shuffle:
            random.shuffle(files)
        for traj in files:
            '''
            new code
            task_queue.qsize()
            '''
            for task_type in task_types:
                if task_type in traj['task']:
                    task_queue.put(traj)
                    break
        return task_queue

    def spawn_threads(self):
        '''
        spawn multiple threads to run eval in parallel
        '''
        task_queue = self.queue_tasks()

        # start threads
        threads = []
        lock = self.manager.Lock()

        self.run(self.model, self.resnet, task_queue, self.args, lock, self.successes, self.failures, self.results)
        # for n in range(self.args.num_threads):
        #     thread = mp.Process(target=self.run, args=(self.model, self.resnet, task_queue, self.args, lock,
        #                                                self.successes, self.failures, self.results))
        #     thread.start()
        #     threads.append(thread)

        # for t in threads:
        #     t.join()

        # save
        self.save_results()

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

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)

    @classmethod
    def run(cls, model, resnet, task_queue, args, lock, successes, failures):
        raise NotImplementedError()

    @classmethod
    def evaluate(cls, env, model, r_idx, resnet, traj_data, args, lock, successes, failures):
        raise NotImplementedError()

    def save_results(self):
        raise NotImplementedError()

    def create_stats(self):
        raise NotImplementedError()

    def _get_openable_points(cls, traj_data):
        scene_num = traj_data['scene']['scene_num']
        openable_json_file = os.path.join(os.environ['ALFRED_ROOT'], 'gen/layouts/FloorPlan%d-openable.json' % scene_num)
        with open(openable_json_file, 'r') as f:
            openable_points = json.load(f)
        return openable_points

    def explore_scene(cls, env, traj_data, resnet, eval_debug, task_goal):
        '''
        Use pre-computed openable points from ALFRED to store receptacle locations
        '''
        thirdpart_meta_data = {
            "agent_sgg_meta_data": None,
            "all_meta_data": None,
            "all_meta_data_1": None,
            "all_meta_data_2": None,
        }
        agent = {
            "exploration_agent_sgg_meta_data": [],
        }
        meta_datas = {
            "exploration_sgg_meta_data": [],
        }
        meta_datas_1 = {
            "exploration_sgg_meta_data": [],
        }
        meta_datas_2 = {
            "exploration_sgg_meta_data": [],
        }
        openable_points = cls._get_openable_points(cls, traj_data)
        agent_height = env.last_event.metadata['agent']['position']['y']
        for recep_id, point in openable_points.items():
            recep_class = recep_id.split("|")[0]
            action = {'action': 'TeleportFull',
                      'x': point[0],
                      'y': agent_height,
                      'z': point[1],
                      'rotateOnTeleport': False,
                      'rotation': point[2],
                      'horizon': point[3]}
            event = env.step(action)
            if event.metadata['lastActionSuccess']:
                agent["exploration_agent_sgg_meta_data"].append(env.last_event.metadata['agent'])
                # 0
                np_image = np.uint8(env.last_event.frame)
                curr_image = Image.fromarray(np_image)
                image_feature = resnet.featurize([curr_image], batch=1)[0]
                meta_data = env.last_event.metadata['objects']
                meta_data = {
                    "exploration_img": image_feature,
                    "exploration_sgg_meta_data": meta_data,
                }
                meta_datas["exploration_sgg_meta_data"].append(meta_data)
                # 1 & 2
                thirdParty_meta_data = [meta_datas_1, meta_datas_2]
                for third_id, dict_image in enumerate(event.third_party_camera_frames):
                    # print(event.metadata['thirdPartyCameras'][third_id])
                    np_image = np.uint8(dict_image["frame"])
                    curr_image = Image.fromarray(np_image)
                    image_feature = resnet.featurize([curr_image], batch=1)[0]
                    meta_data = dict_image['objects']
                    meta_data = {
                        "exploration_img": image_feature,
                        "exploration_sgg_meta_data": meta_data,
                    }
                    thirdParty = thirdParty_meta_data[third_id]
                    thirdParty["exploration_sgg_meta_data"].append(meta_data)
                # eval_debug
                dict_action = {
                    'action_low': ".",
                    'action_navi_low': ".",
                    'action_operation_low': ".",
                    'action_navi_or_operation': [],
                    # goal
                    'subgoal_t': ".",
                    'progress_t': ".",
                    'global_graph_dict_ANALYZE_GRAPH': {},
                    'current_state_dict_ANALYZE_GRAPH': {},
                    'history_changed_dict_ANALYZE_GRAPH': {},
                    'priori_dict_ANALYZE_GRAPH': {},
                    "mask": None,
                    "pred_class": 0,
                    "object": "0",
                }
                eval_debug.add_data(0, np_image, np_image[:, :, 0], dict_action, task_goal, "")
        thirdpart_meta_data["agent_sgg_meta_data"] = agent
        thirdpart_meta_data["all_meta_data"] = meta_datas
        thirdpart_meta_data["all_meta_data_1"] = meta_datas_1
        thirdpart_meta_data["all_meta_data_2"] = meta_datas_2
        return thirdpart_meta_data

    def get_meta_datas(cls, env, resnet):
        thirdpart_meta_data = {
            "agent_sgg_meta_data": None,
            "all_meta_data": {},
            "all_meta_data_1": {},
            "all_meta_data_2": {},
        }
        # get 0
        curr_image = Image.fromarray(np.uint8(env.last_event.frame))
        image_feature = resnet.featurize([curr_image], batch=1)[0]
        meta_datas = {
            "sgg_meta_data":
                    {"rgb_image": image_feature,
                     "sgg_meta_data": env.last_event.metadata['objects']}
        }
        # 1 & 2
        meta_datas_1 = {}
        meta_datas_2 = {}
        thirdParty_meta_data = [meta_datas_1, meta_datas_2]
        for third_id, dict_image in enumerate(env.last_event.third_party_camera_frames):
            # print(event.metadata['thirdPartyCameras'][third_id])
            np_image = np.uint8(dict_image["frame"])
            curr_image = Image.fromarray(np_image)
            image_feature = resnet.featurize([curr_image], batch=1)[0]
            thirdParty = thirdParty_meta_data[third_id]
            thirdParty["sgg_meta_data"] = {
                     "rgb_image": image_feature,
                     "sgg_meta_data": dict_image['objects']}
        # merge
        thirdpart_meta_data["agent_sgg_meta_data"] = {"agent_sgg_meta_data": [env.last_event.metadata['agent']]}
        thirdpart_meta_data["all_meta_data"] = meta_datas
        thirdpart_meta_data["all_meta_data_1"] = meta_datas_1
        thirdpart_meta_data["all_meta_data_2"] = meta_datas_2
        return [thirdpart_meta_data]

    def get_frame_feat(cls, env, resnet, feat):
        last_event = env.last_event
        last_third_party_1 = last_event.third_party_camera_frames[0]
        last_third_party_2 = last_event.third_party_camera_frames[1]

        feat['frames_conv'], feat['frames_instance_conv'], feat['frames_depth_conv'] = \
            cls.process_image(
                cls, resnet, last_event.frame, last_event.instance_segmentation_frame, last_event.depth_frame)
        feat['frames_conv_1'], feat['frames_instance_conv_1'], feat['frames_depth_conv_1'] = \
            cls.process_image(
                cls, resnet, last_third_party_1['frame'], last_third_party_1['instance_segmentation_frame'], last_third_party_1['depth_frame'])
        feat['frames_conv_2'], feat['frames_instance_conv_2'], feat['frames_depth_conv_2'] = \
            cls.process_image(
                cls, resnet, last_third_party_2['frame'], last_third_party_2['instance_segmentation_frame'], last_third_party_2['depth_frame'])
        return feat

    def process_image(cls, resnet, frames, frames_instance, frames_depth):
        frames = Image.fromarray(np.uint8(frames))
        frames = resnet.featurize([frames], batch=1).unsqueeze(0)
        frames_instance = Image.fromarray(np.uint8(frames_instance))
        frames_instance = resnet.featurize([frames_instance], batch=1).unsqueeze(0)
        frames_depth = Image.fromarray(np.uint8(frames_depth * (255 / 10000))).convert('RGB')
        frames_depth = resnet.featurize([frames_depth], batch=1).unsqueeze(0)
        return frames, frames_instance, frames_depth