import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import json
import glob
import os
import constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
from gen.utils.video_util import VideoSaver
from gen.utils.py_util import walklevel
from env.thor_env import ThorEnv


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
OBJECT_META_FOLDER = "sgg_meta"
EXPLORATION_META_FOLDER = "exploration_meta"

SAVE_instance_detections2D = False
INSTANCE_DETECTIONS2D_FILE = "instance_detections2D"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

video_saver = VideoSaver()

fail_log = open("fail_log.txt", "w")

meta_datas = {
    "instance_detections2D": [],
    "instance_detections2D_exp": [],
}


def get_openable_points(traj_data):
    scene_num = traj_data['scene']['scene_num']
    openable_json_file = os.path.join(os.environ['ALFRED_ROOT'], 'gen/layouts/FloorPlan%d-openable.json' % scene_num)
    with open(openable_json_file, 'r') as f:
        openable_points = json.load(f)
    return openable_points


def explore_scene(env, traj_data, root_dir):
    '''
    Use pre-computed openable points from ALFRED to store receptacle locations
    '''
    openable_points = get_openable_points(traj_data)
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
        save_frame(env, event, root_dir, folder_name=EXPLORATION_META_FOLDER)


# class_detections2D
def save_frame(env, event, root_dir, task_desc='None', folder_name=OBJECT_META_FOLDER):
    if SAVE_instance_detections2D:
        global meta_datas
        instance_detections2D = env.last_event.instance_detections2D
        if folder_name == OBJECT_META_FOLDER:
            meta_datas['instance_detections2D'].append(instance_detections2D)
        elif folder_name == EXPLORATION_META_FOLDER:
            meta_datas['instance_detections2D_exp'].append(instance_detections2D)
        else:
            raise NotImplementedError()
    else:
        meta_path = os.path.join(root_dir, folder_name)
        # EXPLORATION_IMG
        if folder_name == EXPLORATION_META_FOLDER:
            im_ind = get_image_index(meta_path)
            rgb_image = event.frame[:, :, ::-1]
            cv2.imwrite(meta_path + '/%09d.png' % im_ind, rgb_image)
        # META DATA
        im_idx = get_json_index(meta_path)
        # store color to object type dictionary
        all_objects = env.last_event.metadata['objects']
        # save sgg meta
        sgg_meta_file = os.path.join(root_dir, folder_name, "%09d.json" % (im_idx))
        with open(sgg_meta_file, 'w') as f:
            json.dump(all_objects, f)
        # print("Total Size: %s" % im_idx)


def get_json_index(save_path):
    file = glob.glob(save_path + '/*.json')
    return len(file)


def get_image_index(save_path):
    max_img = max(len(glob.glob(save_path + '/*.png')), len(glob.glob(save_path + '/*.jpg')))
    return max_img


def save_image_with_delays(env, action,
                           save_path, direction=constants.BEFORE):
    im_ind = get_json_index(save_path)
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        save_frame(env, env.last_event, save_path)
        env.noop()
    return im_ind


def save_images_in_events(env, events, root_dir):
    for event in events:
        save_frame(env, event, root_dir)


def check_dir(path):
    if os.path.exists(path):
        return True
    os.mkdir(path)
    return False


def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # make directories
    root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

    orig_images_dir = os.path.join(root_dir, ORIGINAL_IMAGES_FORLDER)
    object_meta_dir = os.path.join(root_dir, OBJECT_META_FOLDER)
    exploration_dir = os.path.join(root_dir, EXPLORATION_META_FOLDER)
    global meta_datas
    print("SAVE_instance_detections2D set {}".format(SAVE_instance_detections2D))
    meta_datas = {
        "instance_detections2D": [],
        "instance_detections2D_exp": [],
    }
    # fresh images list
    traj_data['images'] = list()

    # clear_and_create_dir(object_meta_dir)
    # clear_and_create_dir(exploration_dir)
    print("no clear_and_create_dir")

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    print(object_meta_dir)
    explore_scene(env, traj_data, root_dir)
    env.step(dict(traj_data['scene']['init_action']))
    # print("Task: %s" % (traj_data['template']['task_desc']))

    # setup task
    env.set_task(traj_data, args, reward_type='dense')
    rewards = []

    for ll_idx, ll_action in enumerate(traj_data['plan']['low_actions']):
        # next cmd under the current hl_action
        cmd = ll_action['api_action']
        hl_action = traj_data['plan']['high_pddl'][ll_action['high_idx']]

        # remove unnecessary keys
        cmd = {k: cmd[k] for k in ['action', 'objectId', 'receptacleObjectId', 'placeStationary', 'forceAction'] if k in cmd}

        if "MoveAhead" in cmd['action']:
            if args.smooth_nav:
                save_frame(env, env.last_event, root_dir)
                events = env.smooth_move_ahead(cmd, render_settings)
                save_images_in_events(env, events, root_dir)
                event = events[-1]
            else:
                save_frame(env, env.last_event, root_dir)
                event = env.step(cmd)

        elif "Rotate" in cmd['action']:
            if args.smooth_nav:
                save_frame(env, env.last_event, root_dir)
                events = env.smooth_rotate(cmd, render_settings)
                save_images_in_events(env, events, root_dir)
                event = events[-1]
            else:
                save_frame(env, env.last_event, root_dir)
                event = env.step(cmd)

        elif "Look" in cmd['action']:
            if args.smooth_nav:
                save_frame(env, env.last_event, root_dir)
                events = env.smooth_look(cmd, render_settings)
                save_images_in_events(env, events, root_dir)
                event = events[-1]
            else:
                save_frame(env, env.last_event, root_dir)
                event = env.step(cmd)

        # handle the exception for CoolObject tasks where the actual 'CoolObject' action is actually 'CloseObject'
        # TODO: a proper fix for this issue
        elif "CloseObject" in cmd['action'] and \
             "CoolObject" in hl_action['planner_action']['action'] and \
             "OpenObject" in traj_data['plan']['low_actions'][ll_idx + 1]['api_action']['action']:
            if args.time_delays:
                cool_action = hl_action['planner_action']
                save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.BEFORE)
                event = env.step(cmd)
                save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.MIDDLE)
                save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.AFTER)
            else:
                save_frame(env, env.last_event, root_dir)
                event = env.step(cmd)

        else:
            if args.time_delays:
                save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.BEFORE)
                event = env.step(cmd)
                save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.MIDDLE)
                save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.AFTER)
            else:
                save_frame(env, env.last_event, root_dir)
                event = env.step(cmd)

        if not event.metadata['lastActionSuccess']:
            print("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))
            fail_log.write("Replay Failed: %s \n" % (env.last_event.metadata['errorMessage']))
            raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))

        reward, _ = env.get_transition_reward()
        rewards.append(reward)

    # save 10 frames in the end as per the training data
    for _ in range(10):
        save_frame(env, env.last_event, root_dir)

    # check if number of new images is the same as the number of original images
    if args.smooth_nav and args.time_delays:
        orig_img_count = get_image_index(orig_images_dir)
        object_meta_count = get_json_index(object_meta_dir)
        if SAVE_instance_detections2D:
            object_meta_count = len(meta_datas["instance_detections2D"])
            instance_detection_file = os.path.join(root_dir, "INSTANCE_DETECTIONS2D_FILE.json")
            with open(instance_detection_file, 'w') as f:
                json.dump(meta_datas, f)
        print ("Original Image Count %d, New Image Count %d" % (orig_img_count, object_meta_count))
        if orig_img_count != object_meta_count:
            print("sequence length doesn't match\n" + object_meta_dir + "\n")
            fail_log.write("sequence length doesn't match\n" + object_meta_dir + "\n")
            fail_log.write("Original Image Count %d, New Image Count %d" % (orig_img_count, object_meta_count))
            raise Exception("WARNING: the augmented sequence length doesn't match the original")


def run():
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,
                  player_screen_height=IMAGE_HEIGHT)

    skipped_files = []

    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print ("Augmenting: " + json_file)
        try:
            augment_traj(env, json_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print ("Error: " + repr(e))
            print ("Skipping " + json_file)
            skipped_files.append(json_file)
            fail_log.write(repr(e) + "\n")
            fail_log.write(json_file + "\n")

    env.stop()
    print("Finished.")

    # skipped files
    if len(skipped_files) > 0:
        print("Skipped Files:")
        print(skipped_files)


traj_list = []
lock = threading.Lock()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data/2.1.0")
parser.add_argument('--split', type=str, default='valid_seen', choices=['train', 'valid_seen', 'valid_unseen'])
parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true')
parser.add_argument('--time_delays', dest='time_delays', action='store_true')
parser.add_argument('--shuffle', dest='shuffle', action='store_true')
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--reward_config', type=str, default='../models/config/rewards.json')
args = parser.parse_args()

# make a list of all the traj_data json files
for split in ['train/', 'valid_seen/', 'valid_unseen/']:
    for dir_name, subdir_list, file_list in walklevel(args.data_path + split, level=2):
        if "trial_" in dir_name:
            json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
            # import pdb; pdb.set_trace()
            if not os.path.isfile(json_file):
                continue
            traj_list.append(json_file)
# traj_list = ['../data/full_2.1.0/train/pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T20190907_151643_465634/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-DiningTable-24/trial_T20190908_194409_961394/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-AppleSliced-None-DiningTable-27/trial_T20190907_171803_405680/traj_data.json', '../data/full_2.1.0/train/pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-14/trial_T20190910_120350_730711/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-Lettuce-None-SinkBasin-23/trial_T20190908_173530_026785/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879/traj_data.json', '../data/full_2.1.0/train/pick_heat_then_place_in_recep-AppleSliced-None-SideTable-3/trial_T20190908_110347_206140/traj_data.json', '../data/full_2.1.0/train/pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-GarbageCan-11/trial_T20190909_013637_168506/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572/traj_data.json', '../data/full_2.1.0/train/look_at_obj_in_light-Pen-None-DeskLamp-316/trial_T20190908_061814_700195/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-PotatoSliced-None-CounterTop-19/trial_T20190909_053101_102010/traj_data.json', '../data/full_2.1.0/train/look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182531_510491/traj_data.json', '../data/full_2.1.0/train/look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182720_056041/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-LettuceSliced-Pot-DiningTable-21/trial_T20190907_160923_689765/traj_data.json', '../data/full_2.1.0/train/look_at_obj_in_light-Pillow-None-DeskLamp-319/trial_T20190907_224211_927258/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-LettuceSliced-None-GarbageCan-6/trial_T20190907_210244_406018/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-26/trial_T20190908_162237_908840/traj_data.json', '../data/full_2.1.0/train/pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-Pen-Bowl-Dresser-311/trial_T20190908_170820_174380/traj_data.json', '../data/full_2.1.0/train/pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674/traj_data.json', '../data/full_2.1.0/train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433/traj_data.json', '../data/full_2.1.0/train/pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614/traj_data.json', '../data/full_2.1.0/train/pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-23/trial_T20190907_123248_978930/traj_data.json', ]
# random shuffle
if args.shuffle:
    random.shuffle(traj_list)
# start threads
# run()
threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run)
    threads.append(thread)
    thread.start()
    time.sleep(1)