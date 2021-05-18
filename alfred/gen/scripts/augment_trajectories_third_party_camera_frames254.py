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
from env.thor_env_gen_254 import ThorEnv
# event.third_party_camera_frames

TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
THIRDPARTYCAMERAS_IMAGES_FORLDER = ["raw_images_1", "raw_images_2"]

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

video_saver = VideoSaver()

fail_log = open("fail_log.txt", "w")


def get_image_index(save_path):
    max_img = max(len(glob.glob(save_path + '/*.png')), len(glob.glob(save_path + '/*.jpg')))
    return max_img


def save_image_with_delays(env, action,
                           save_path, direction=constants.BEFORE):
    im_ind = get_image_index(save_path)
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        save_image(env.last_event, save_path)
        env.noop()
    return im_ind


def save_image(event, save_path):
    # dump images
    thirdPartyCameras = [os.path.join(save_path, THIRDPARTYCAMERAS_IMAGES_FORLDER[0]), os.path.join(save_path, THIRDPARTYCAMERAS_IMAGES_FORLDER[1])]
    for third_id, image in enumerate(event.third_party_camera_frames):
        # print(event.metadata['thirdPartyCameras'][third_id])
        path = thirdPartyCameras[third_id]
        im_ind = get_image_index(path)
        image = image[:, :, ::-1]
        cv2.imwrite(path + '/%09d.png' % (im_ind), image)

    return im_ind


def save_images_in_events(events, root_dir):
    for event in events:
        save_image(event, root_dir)


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
    thirdparty_images_dir_1 = os.path.join(root_dir, THIRDPARTYCAMERAS_IMAGES_FORLDER[0])
    thirdparty_images_dir_2 = os.path.join(root_dir, THIRDPARTYCAMERAS_IMAGES_FORLDER[1])

    # fresh images list
    traj_data['images'] = list()
    print("get_image_index(orig_images_dir): ", get_image_index(orig_images_dir))
    print("get_image_index(thirdparty_images_dir_1): ", get_image_index(thirdparty_images_dir_1))
    if get_image_index(orig_images_dir) == 0:
        print("get_image_index(orig_images_dir) == 0" + orig_images_dir + "\n")
        fail_log.write("get_image_index(orig_images_dir) == 0" + orig_images_dir + "\n")
        return
    elif get_image_index(orig_images_dir) == get_image_index(thirdparty_images_dir_1) \
       and get_image_index(orig_images_dir) == get_image_index(thirdparty_images_dir_2):
        print("already create: " + orig_images_dir + "\n")
        fail_log.write("already create: " + orig_images_dir + "\n")
        return
    clear_and_create_dir(thirdparty_images_dir_1)
    clear_and_create_dir(thirdparty_images_dir_2)

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    # FloorPlan308
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    print(thirdparty_images_dir_1)
    env.step(**dict(traj_data['scene']['init_action']))
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
                save_image(env.last_event, root_dir)
                events = env.smooth_move_ahead(render_settings, **cmd)
                save_images_in_events(events, root_dir)
                event = events[-1]
            else:
                save_image(env.last_event, root_dir)
                event = env.step(**cmd)

        elif "Rotate" in cmd['action']:
            if args.smooth_nav:
                save_image(env.last_event, root_dir)
                events = env.smooth_rotate(render_settings, **cmd)
                save_images_in_events(events, root_dir)
                event = events[-1]
            else:
                save_image(env.last_event, root_dir)
                event = env.step(**cmd)

        elif "Look" in cmd['action']:
            if args.smooth_nav:
                save_image(env.last_event, root_dir)
                events = env.smooth_look(render_settings, **cmd)
                save_images_in_events(events, root_dir)
                event = events[-1]
            else:
                save_image(env.last_event, root_dir)
                event = env.step(**cmd)

        # handle the exception for CoolObject tasks where the actual 'CoolObject' action is actually 'CloseObject'
        # TODO: a proper fix for this issue
        elif "CloseObject" in cmd['action'] and \
             "CoolObject" in hl_action['planner_action']['action'] and \
             "OpenObject" in traj_data['plan']['low_actions'][ll_idx + 1]['api_action']['action']:
            if args.time_delays:
                cool_action = hl_action['planner_action']
                save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.BEFORE)
                event = env.step(**cmd)
                save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.MIDDLE)
                save_image_with_delays(env, cool_action, save_path=root_dir, direction=constants.AFTER)
            else:
                save_image(env.last_event, root_dir)
                event = env.step(**cmd)

        else:
            if args.time_delays:
                save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.BEFORE)
                event = env.step(**cmd)
                save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.MIDDLE)
                save_image_with_delays(env, cmd, save_path=root_dir, direction=constants.AFTER)
            else:
                save_image(env.last_event, root_dir)
                event = env.step(**cmd)

        # update image list
        new_img_idx = get_image_index(thirdparty_images_dir_1)
        last_img_idx = len(traj_data['images'])
        num_new_images = new_img_idx - last_img_idx
        for j in range(num_new_images):
            traj_data['images'].append({
                'low_idx': ll_idx,
                'high_idx': ll_action['high_idx'],
                'image_name': '%09d.png' % int(last_img_idx + j)
            })

        if not event.metadata['lastActionSuccess']:
            print("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))
            fail_log.write("Replay Failed: %s \n" % thirdparty_images_dir_1)
            # fail_log.write("Replay Failed: %s \n" % (env.last_event.metadata['errorMessage']))
            # raise Exception("Replay Failed: %s" % (env.last_event.metadata['errorMessage']))
            return

        reward, _ = env.get_transition_reward()
        rewards.append(reward)

    # save 10 frames in the end as per the training data
    for _ in range(10):
        save_image(env.last_event, root_dir)

    # save video
    images_path = os.path.join(thirdparty_images_dir_1, '*.png')

    # check if number of new images is the same as the number of original images
    if args.smooth_nav and args.time_delays:
        orig_img_count = get_image_index(orig_images_dir)
        new_img_count = get_image_index(thirdparty_images_dir_1)
        print ("Original Image Count %d, New Image Count %d" % (orig_img_count, new_img_count))
        video_save_path = os.path.join(thirdparty_images_dir_1, 'high_res_video.mp4')
        video_saver.save(images_path, video_save_path)
        if orig_img_count != new_img_count:
            print("sequence length doesn't match\n" + thirdparty_images_dir_1 + "\n")
            fail_log.write("sequence length doesn't match\n" + thirdparty_images_dir_1 + "\n")
            fail_log.write("Original Image Count %d, New Image Count %d" % (orig_img_count, new_img_count))
            # raise Exception("WARNING: the augmented sequence length doesn't match the original")
            return


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
            # fail_log.write(repr(e) + "\n")
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
traj_list = ['../data/full_2.1.0/valid_unseen/look_at_obj_in_light-Book-None-DeskLamp-308/trial_T20190908_144951_587345/traj_data.json', '../data/full_2.1.0/valid_unseen/pick_cool_then_place_in_recep-Mug-None-Cabinet-10/trial_T20190909_121635_622676/traj_data.json']
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