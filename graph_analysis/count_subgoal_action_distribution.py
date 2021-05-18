from collections import defaultdict
import argparse
import numpy as np
import shutil
import glob
import json
import os
import sys
TRAJ_DATA_JSON_FILENAME = "traj_data.json"
OUTPUT_SUB_GOAL_DISTRIBUTION = "data_dgl/subgoal_distribution.json"
traj_list = []


def get_subgoal_lowaction_define():
    D_SUB_GOAL = ["GotoLocation", "PickupObject", "PutObject", "CoolObject",
                  "HeatObject", "CleanObject", "SliceObject", "ToggleObject"]
    D_LOW_ACT_NAMES = ["LookDown", "MoveAhead", "RotateLeft", "LookUp", "RotateRight",
                       "PickupObject", "SliceObject", "OpenObject", "PutObject",
                       "CloseObject", "ToggleObjectOn", "ToggleObjectOff",
                       ]
    return D_SUB_GOAL, D_LOW_ACT_NAMES


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


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


def analize_traj(json_file):
    '''
    "plan": {
        "high_pddl": [
            {
                "discrete_action": {
                    "action": "GotoLocation",
                    "args": [
                        "shelf"
                    ]
                },
                "high_idx": 0,
                "planner_action": {
                    "action": "GotoLocation",
                    "location": "loc|0|-6|2|60"
                }
            },
        "low_actions": [
            {
                "api_action": {
                    "action": "LookDown",
                    "forceAction": true
                },
                "discrete_action": {
                    "action": "LookDown_15",
                    "args": {}
                },
                "high_idx": 0
            },
    '''
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)
    # import pdb; pdb.set_trace()
    list_high_sg = traj_data["plan"]["high_pddl"]
    list_low_act = traj_data["plan"]["low_actions"]
    dict_ind_sg = {}
    for sg in list_high_sg:
        int_high_idx = sg["high_idx"]
        string_sg = sg["discrete_action"]["action"]
        dict_ind_sg[int_high_idx] = string_sg
        if string_sg in sg_distribution:
            sg_distribution[string_sg]["total_num"] += 1
    for low_act in list_low_act:
        int_high_idx = low_act["high_idx"]
        string_sg = dict_ind_sg[int_high_idx]
        act_name = low_act["api_action"]["action"]
        sg_distribution[string_sg][act_name] += 1
    with open(OUTPUT_SUB_GOAL_DISTRIBUTION, "w") as f:
        json.dump(sg_distribution, f)


def run():
    skipped_files = []
    num_total_traj = len(traj_list)

    while len(traj_list) > 0:
        json_file = traj_list.pop()
        # make directories
        analize_traj(json_file)

    print(sg_distribution)
    print("total traj num: {}".format(num_total_traj))
    # skipped files
    print(skipped_files)
    print("Skipped Files: {}".format(len(skipped_files)))


sub_goal, low_act_names = get_subgoal_lowaction_define()
sg_distribution = {}
for sg in sub_goal:
    sg_distribution[sg] = dict()
    sg_distribution[sg]["total_num"] = 0
    for act_name in low_act_names:
        sg_distribution[sg][act_name] = 0
if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="../alfred/data/json_feat_2.1.0/")
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
