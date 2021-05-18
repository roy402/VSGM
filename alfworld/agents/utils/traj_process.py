import os
import cv2
import json
import numpy as np
import h5py
from PIL import Image
TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


def save_trajectory(envs, store_states, task_desc_strings, expert_actions, still_running_masks):
    print("=== SAVE BATCH ===")
    TRAIN_DATA = "TRAIN_DATA.json"
    for i, thor in enumerate(envs):
        save_data_path = thor.env.save_frames_path
        print("=== save one episode len ===", len(expert_actions))
        print("=== save path ===", save_data_path)
        data = {
            "task_desc_string": [],
            "expert_action": [],
            "sgg_meta_data": [],
            "rgb_image": [],
        }
        img_name = 0
        for store_state, task_desc_string, expert_action, still_running_mask in \
            zip(store_states, task_desc_strings, expert_actions, still_running_masks):
            if int(still_running_mask[i]) == 0:
                break
            _task_desc_string = task_desc_string[i]
            _expert_action = expert_action[i]
            rgb_image = store_state[i]["rgb_image"]
            img_path = os.path.join(save_data_path, '%09d.png' % img_name)
            cv2.imwrite(img_path, rgb_image)

            data["task_desc_string"].append(_task_desc_string)
            data["expert_action"].append(_expert_action)
            data["rgb_image"].append(img_path)
            data["sgg_meta_data"].append(store_state[i]["sgg_meta_data"])
            img_name += 1

        with open(os.path.join(save_data_path, TRAIN_DATA), 'w') as f:
            json.dump(data, f)


def save_exploration_trajectory(envs, exploration_frames, sgg_meta_datas):
    print("=== SAVE EXPLORATION BATCH ===")
    TRAIN_DATA = "TRAIN_DATA.json"
    for i, thor in enumerate(envs):
        save_data_path = thor.env.save_frames_path
        print("=== save exploration one episode len ===", len(sgg_meta_datas[i]))
        print("=== save exploration path ===", save_data_path)
        data = {
            "exploration_img": [],
            "exploration_sgg_meta_data": [],
        }
        img_name = 0
        for exploration_frame, sgg_meta_data, in zip(exploration_frames[i], sgg_meta_datas[i]):
            img_path = os.path.join(save_data_path, 'exploration_img%09d.png' % img_name)
            cv2.imwrite(img_path, exploration_frame)

            data["exploration_img"].append(img_path)
            data["exploration_sgg_meta_data"].append(sgg_meta_data)
            img_name += 1
        with open(os.path.join(save_data_path, TRAIN_DATA), 'r') as f:
            ori_data = json.load(f)
        with open(os.path.join(save_data_path, TRAIN_DATA), 'w') as f:
            data = {**ori_data, **data}
            json.dump(data, f)


def get_traj_train_data(tasks_paths, save_frames_path):
    # [store_states, task_desc_strings, expert_actions]
    transition_caches = []
    for task_path in tasks_paths:
        transition_cache = [None, None, None]
        traj_root = os.path.dirname(task_path)
        task_path = os.path.join(save_frames_path, traj_root.replace('../', ''))
        with open(task_path + '/TRAIN_DATA.json', 'r') as f:
            data = json.load(f)
        # store store_states
        store_states = []
        rgb_array = load_img_with_h5(data["rgb_image"], task_path)
        for img, sgg_meta_data in zip(rgb_array, data["sgg_meta_data"]):
            store_state = {
                "rgb_image": img,
                "sgg_meta_data": sgg_meta_data,
            }
            store_states.append(store_state)
        # len(store_state) == 39
        transition_cache[0] = store_states
        # len(seq_task_desc_strings) == 39
        transition_cache[1] = [[task_desc_string] for task_desc_string in data["task_desc_string"]]
        # len(seq_target_strings) == 39
        transition_cache[2] = [[expert_action] for expert_action in data["expert_action"]]
        transition_caches.append(transition_cache)
    # import pdb; pdb.set_trace()
    return transition_caches


def get_exploration_traj_train_data(tasks_paths, save_frames_path):
    # [store_states, task_desc_strings, expert_actions]
    exploration_transition_caches = []
    for task_path in tasks_paths:
        transition_cache = [None, None, None]
        traj_root = os.path.dirname(task_path)
        task_path = os.path.join(save_frames_path, traj_root.replace('../', ''))
        with open(task_path + '/TRAIN_DATA.json', 'r') as f:
            data = json.load(f)
        # store store_states
        store_states = []
        rgb_array = load_img_with_h5(data["exploration_img"], task_path, pt_name="exploration_img.pt")
        for img, sgg_meta_data in zip(rgb_array, data["exploration_sgg_meta_data"]):
            store_state = {
                "exploration_img": img,
                "exploration_sgg_meta_data": sgg_meta_data,
            }
            store_states.append(store_state)
        # len(store_state) == 39
        transition_cache[0] = store_states
        exploration_transition_caches.append(transition_cache)
    # import pdb; pdb.set_trace()
    return exploration_transition_caches


def load_img_with_h5(rgb_img_names, img_dir_path, pt_name="img.pt"):
    img_h5 = os.path.join(img_dir_path, pt_name)
    if not os.path.isfile(img_h5):
        rgb_array = []
        for rgb_img_name in rgb_img_names:
            rgb_img_name = rgb_img_name.rsplit("/", 1)[-1]
            rgb_img_path = os.path.join(img_dir_path, rgb_img_name)
            rgb_img = Image.open(rgb_img_path).convert("RGB")
            rgb_img = np.array(rgb_img)
            rgb_array.append(rgb_img)
        hf = h5py.File(img_h5, 'w')
        hf.create_dataset('rgb_array', data=rgb_array)
        hf.close()
        print("Save img data to {}".format(img_h5))
    hf = h5py.File(img_h5, 'r')
    rgb_array = hf['rgb_array'][:]
    return rgb_array
