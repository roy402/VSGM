# https://ai2thor.allenai.org/ithor/documentation/metadata/
# event.metadata['objects']

# #example return of object metadata for a single sim object of type `Box`
# [{'name': 'DeskLamp_33ba15a6',
#   'position': {'x': -1.31978738, 'y': 1.23870516, 'z': -0.994436145},
#   'rotation': {'x': 359.900757, 'y': 89.95341, 'z': 359.986816},
#   'visible': False,
#   'receptacle': False,
#   'toggleable': True,
#   'isToggled': True,
#   'breakable': False,
#   'isBroken': False,
#   'canFillWithLiquid': False,
#   'isFilledWithLiquid': False,
#   'dirtyable': False,
#   'isDirty': False,
#   'canBeUsedUp': False,
#   'isUsedUp': False,
#   'cookable': False,
#   'isCooked': False,
#   'ObjectTemperature': 'RoomTemp',
#   'canChangeTempToHot': False,
#   'canChangeTempToCold': False,
#   'sliceable': False,
#   'isSliced': False,
#   'openable': False,
#   'isOpen': False,
#   'pickupable': False,
#   'isPickedUp': False,
#   'moveable': True,
#   'mass': 2.06,
#   'salientMaterials': ['Metal', 'Fabric'],
#   'receptacleObjectIds': None,
#   'distance': 4.10230827,
#   'objectType': 'DeskLamp',
#   'objectId': 'DeskLamp|-01.32|+01.24|-00.99',
#   'parentReceptacles': ['Dresser|-01.33|+00.01|-00.74'],
#   'isMoving': False,
#   'axisAlignedBoundingBox': {
#     'cornerPoints': [[-1.24880266, 1.56558228, -0.902912259],
#                      [-1.24880266, 1.56558228, -1.086083],
#                      [-1.24880266, 1.23842049, -0.902912259],
#                      [-1.24880266, 1.23842049, -1.086083],
#                      [-1.390816, 1.56558228, -0.902912259],
#                      [-1.390816, 1.56558228, -1.086083],
#                      [-1.390816, 1.23842049, -0.902912259],
#                      [-1.390816, 1.23842049, -1.086083]],
#     'center': {'x': -1.31980932, 'y': 1.40200138, 'z': -0.994497657},
#     'size': {'x': 0.142013311, 'y': 0.3271618, 'z': 0.1831708}
#   },
#   'objectOrientedBoundingBox': {
#      'cornerPoints': [[-1.24066579, 1.22984934, -1.10477746],
#                       [-1.24084544, 1.22990012, -0.8839622],
#                       [-1.39887786, 1.22962642, -0.8840907],
#                       [-1.39869821, 1.22957551, -1.104906],
#                       [-1.241268, 1.57760191, -1.10485792],
#                       [-1.24144769, 1.57765281, -0.88404274],
#                       [-1.39948022, 1.57737911, -0.884171247],
#                       [-1.39930058, 1.57732821, -1.10498643]]
#   }
# }]

import os
import sys
sys.path.append(os.path.join(os.environ['ALFWORLD_ROOT']))
# sys.path.append(os.path.join(os.environ['ALFWORLD_ROOT'], 'gen'))
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'gen'))
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
from utils.video_util import VideoSaver
from utils.py_util import walklevel
from env.thor_env import ThorEnv
import pdb

TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

IMAGES_FOLDER = "images"
MASKS_FOLDER = "masks"
META_FOLDER = "meta"
SGG_META_FOLDER = "sgg_meta"

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 400

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

video_saver = VideoSaver()


def get_image_index(save_path):
    return len(glob.glob(save_path + '/*.png'))


def save_image_with_delays(env, action,
                           save_path, direction=constants.BEFORE):
    im_ind = get_image_index(save_path)
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        save_image(env.last_event, save_path)
        env.noop()
    return im_ind


def save_image(event, save_path):
    # rgb
    rgb_save_path = os.path.join(save_path, IMAGES_FOLDER)
    rgb_image = event.frame[:, :, ::-1]

    # masks
    mask_save_path = os.path.join(save_path, MASKS_FOLDER)
    mask_image = event.instance_segmentation_frame

    # dump images
    im_ind = get_image_index(rgb_save_path)
    cv2.imwrite(rgb_save_path + '/%09d.png' % im_ind, rgb_image)
    cv2.imwrite(mask_save_path + '/%09d.png' % im_ind, mask_image)
    return im_ind


def save_images_in_events(events, root_dir):
    for event in events:
        save_image(event, root_dir)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_scene_type(scene_num):
    if scene_num < 100:
        return 'kitchen'
    elif scene_num < 300:
        return 'living'
    elif scene_num < 400:
        return 'bedroom'
    else:
        return 'bathroom'


def get_openable_points(traj_data):
    scene_num = traj_data['scene']['scene_num']
    openable_json_file = os.path.join(os.environ['ALFWORLD_ROOT'], 'gen/layouts/FloorPlan%d-openable.json' % scene_num)
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
        save_frame(env, event, root_dir)


def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)


    # fresh images list
    traj_data['images'] = list()

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    scene_type = get_scene_type(scene_num)
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    root_dir = os.path.join(args.save_path, scene_type)

    imgs_dir = os.path.join(root_dir, IMAGES_FOLDER)
    mask_dir = os.path.join(root_dir, MASKS_FOLDER)
    meta_dir = os.path.join(root_dir, META_FOLDER)
    sgg_meta_dir = os.path.join(root_dir, SGG_META_FOLDER)

    create_dir(imgs_dir)
    create_dir(mask_dir)
    create_dir(meta_dir)
    create_dir(sgg_meta_dir)

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
            event = env.step(cmd)

        elif "Rotate" in cmd['action']:
            event = env.step(cmd)

        elif "Look" in cmd['action']:
            event = env.step(cmd)

        else:
            event = env.step(cmd)
            save_frame(env, event, root_dir, traj_data['turk_annotations']['anns'][0]['task_desc'])

        if not event.metadata['lastActionSuccess']:
            raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))
    raise


# class_detections2D
def save_frame(env, event, root_dir, task_desc='None'):
    im_idx = save_image(event, root_dir)
    # store color to object type dictionary
    color_to_obj_id_type = {}
    all_objects = env.last_event.metadata['objects']
    # [a['name'] for a in event.metadata['objects']]
    # =
    # [a['name'] for a in env.last_event.metadata['objects']]
    for color, object_id in env.last_event.color_to_object_id.items():
        color_to_obj_id_type[str(color)] = object_id
    meta_file = os.path.join(root_dir, META_FOLDER, "%09d.json" % im_idx)
    with open(meta_file, 'w') as f:
        json.dump(color_to_obj_id_type, f)
    # save sgg meta
    sgg_meta_file = os.path.join(root_dir, SGG_META_FOLDER, "%09d%s.json" % (im_idx, task_desc))
    with open(sgg_meta_file, 'w') as f:
        json.dump(all_objects, f)
    # print("Total Size: %s" % im_idx)


def run():
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,
                  player_screen_height=IMAGE_HEIGHT)

    skipped_files = []
    finished = []
    cache_file = os.path.join(args.save_path, "cache.json")

    while len(traj_list) > 0:
        json_file = traj_list.pop()

        print ("(%d Left) Augmenting: %s" % (len(traj_list), json_file))
        try:
            augment_traj(env, json_file)
            finished.append(json_file)
            with open(cache_file, 'w') as f:
                json.dump({'finished': finished}, f)

        except Exception as e:
                import traceback
                traceback.print_exc()
                print ("Error: " + repr(e))
                print ("Skipping " + json_file)
                skipped_files.append(json_file)

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
parser.add_argument('--save_path', type=str, default="detector/data/")
parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true')
parser.add_argument('--time_delays', dest='time_delays', action='store_true')
parser.add_argument('--shuffle', dest='shuffle', action='store_true')
parser.add_argument('--num_threads', type=int, default=1)
parser.add_argument('--reward_config', type=str, default='agents/config/rewards.json')
args = parser.parse_args()


# cache
cache_file = os.path.join(args.save_path, "cache.json")
if os.path.isfile(cache_file):
    with open(cache_file, 'r') as f:
        finished_jsons = json.load(f)
else:
    finished_jsons = {'finished': []}

# make a list of all the traj_data json files
for dir_name, subdir_list, file_list in walklevel(args.data_path, level=2):
    if "trial_" in dir_name:
        json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
        if not os.path.isfile(json_file) or json_file in finished_jsons['finished']:
            continue
        traj_list.append(json_file)

# random shuffle
if args.shuffle:
    random.shuffle(traj_list)

# start threads
run()
# threads = []
# for n in range(args.num_threads):
#     thread = threading.Thread(target=run)
#     threads.append(thread)
#     thread.start()
#     time.sleep(1)