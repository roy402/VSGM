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
from utils.py_util import walklevel


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
HIGH_RES_IMAGES_FOLDER = "high_res_images"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

skipped_files_log = open("skipped_files.txt", "w")


def get_image_index(save_path):
    max_img = max(len(glob.glob(save_path + '/*.png')), len(glob.glob(save_path + '/*.jpg')))
    return max_img


def check_dir(path):
    if os.path.exists(path):
        return True
    os.mkdir(path)
    return False


def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def run():
    skipped_files = []
    num_total_traj = len(traj_list)

    while len(traj_list) > 0:
        json_file = traj_list.pop()
        # make directories
        root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

        orig_images_dir = os.path.join(root_dir, ORIGINAL_IMAGES_FORLDER)
        depth_images_dir = os.path.join(root_dir, DEPTH_IMAGES_FOLDER)
        if get_image_index(orig_images_dir) != get_image_index(depth_images_dir):
            skipped_files.append(json_file)
            skipped_files_log.write("skipped file: {}\n".format(json_file))

    print("total traj num: {}".format(num_total_traj))
    # skipped files
    print(skipped_files)
    print("Skipped Files: {}".format(len(skipped_files)))


traj_list = []

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="data/2.1.0")
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

# start threads
run()