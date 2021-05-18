import sys
import os
import numpy as np
import torch
import random
import json
import argparse
from PIL import Image
import cv2
import math
from detector import transforms as T
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], '..'))
from alfworld.gen import constants

OBJECTS_DETECTOR = constants.OBJECTS_DETECTOR
# 1 background + 108 object
OBJECTS = constants.OBJECTS
STATIC_RECEPTACLES = constants.STATIC_RECEPTACLES
# 1 background + 105 object
ALL_DETECTOR = constants.ALL_DETECTOR
'''
# relation
# 'objectType': 'DeskLamp',
# 'objectId': 'DeskLamp|-01.32|+01.24|-00.99',
# 'parentReceptacles': ['Dresser|-01.33|+00.01|-00.74'],
'''
RELATION = \
    {'parentReceptacles' : 1
     }
NUM_RELATION = len(RELATION)
'''
# Attribute
'''
ATTRIBUTE = \
    ['visible', 'receptacle', 'toggleable', 'isToggled', 'breakable', 'isBroken',
     'canFillWithLiquid', 'isFilledWithLiquid', 'dirtyable', 'isDirty', 'canBeUsedUp',
     'isUsedUp', 'cookable', 'isCooked', 'canChangeTempToHot', 'canChangeTempToCold',
     'sliceable', 'isSliced', 'openable', 'isOpen', 'pickupable', 'isPickedUp',
     'isMoving']
NUM_ATTRIBUTE = len(ATTRIBUTE)
print('NUM_ATTRIBUTE: ', NUM_ATTRIBUTE)
print('ALL_DETECTOR: ', len(ALL_DETECTOR))
print('ALL_DETECTOR + background: ', len(ALL_DETECTOR) + 1)


def get_object_classes(object_type):
    if object_type == "objects":
        # 73
        return OBJECTS_DETECTOR
    elif object_type == "ori_objects":
        # 108
        return OBJECTS
    elif object_type == "receptacles":
        # 32
        return STATIC_RECEPTACLES
    else:
        # 105
        return ALL_DETECTOR


def get_dict_class_to_ind(classes, ):
    class_to_ind = {}
    for ind, label in enumerate(classes):
        class_to_ind[label] = ind
    return class_to_ind


def get_dict_predicate_to_ind():
    return RELATION


def get_transform(cfg, train):
    # transforms = []
    # transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    # return T.Compose(transforms)
    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    flip_vertical_prob = cfg.INPUT.VERTICAL_FLIP_PROB_TRAIN
    brightness = cfg.INPUT.BRIGHTNESS
    contrast = cfg.INPUT.CONTRAST
    saturation = cfg.INPUT.SATURATION
    hue = cfg.INPUT.HUE
    to_bgr255 = cfg.INPUT.TO_BGR255

    # normalize_transform = T.Normalize(
    #     mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    # )
    color_jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = T.Compose(
        [
            color_jitter,
            # T.Resize(min_size, max_size),
            T.ToTensor(),
            # normalize_transform,
        ]
    )
    return transform


'''
data_obj_relation_attribute : [{objectId, attribute, parentReceptacles}, {objectId, attribute, parentReceptacles},]
'''
def transfer_mask_semantic_to_bbox_label(mask, color_to_object, object_classes, data_obj_relation_attribute, MIN_PIXELS=100):
    im_width, im_height = mask.shape[0], mask.shape[1]
    seg_colors = np.unique(mask.reshape(im_height*im_height, 3), axis=0)

    masks, boxes, boxes_id, labels = [], [], [], []
    for color in seg_colors:
        color_str = str(tuple(color[::-1]))
        if color_str in color_to_object:
            object_id = color_to_object[color_str]
            object_class = object_id.split("|", 1)[0] if "|" in object_id else ""
            if "Basin" in object_id:
                object_class += "Basin"
            if object_class in object_classes:
                smask = np.all(mask == color, axis=2)
                pos = np.where(smask)
                num_pixels = len(pos[0])

                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                # skip if not sufficient pixels
                # if num_pixels < MIN_PIXELS:
                if (xmax-xmin)*(ymax-ymin) < MIN_PIXELS:
                    continue

                class_idx = object_classes.index(object_class)

                masks.append(smask)
                boxes.append([xmin, ymin, xmax, ymax])
                # import pdb; pdb.set_trace()
                labels.append(class_idx)
                boxes_id.append(object_id)

    return np.array(masks), np.array(boxes), np.array(labels), boxes_id


def transfer_object_meta_data_to_relation_and_attribute(boxes_id, data_obj_relation_attribute, horizontal_view_angle, agent_meta):
    # relation & attribute
    obj_relations, obj_relation_triplets, obj_attributes, obj_angle_of_view = [], [], [], []
    obj_relations = np.zeros((len(boxes_id), len(boxes_id)))


    for i, object_id in enumerate(boxes_id):
        # {objectId, attribute, parentReceptacles}
        for obj_relation_attribute in data_obj_relation_attribute:
            if obj_relation_attribute['objectId'] == object_id:
                obj_attribute = _set_obj_attribute(obj_relation_attribute)
                parentReceptacles_ids = obj_relation_attribute['parentReceptacles']
                if parentReceptacles_ids and len(parentReceptacles_ids) > 0:
                    for parentReceptacles_id in parentReceptacles_ids:
                        '''
                        relation
                         Sofa|-02.96|+00.08|+01.39
                        Pillow|-02.89|+00.62|+00.82
                        relation
                         Sofa|-02.96|+00.08|+01.39
                        RemoteControl|-03.03|+00.56|+02.01
                        relation
                         Sofa|-02.96|+00.08|+01.39
                        Laptop|-02.81|+00.56|+01.81
                        relation
                         Sofa|-02.96|+00.08|+01.39
                        Pillow|-02.89|+00.62|+01.19
                        '''
                        j, relation, isfind = _search_boxes_index(boxes_id, parentReceptacles_id)
                        # print(object_id)
                        if isfind:
                            obj_relations[j, i] = relation
                            obj_relation_triplets.append(np.array([j, i, relation]))
                obj_attributes.append(obj_attribute)
                angle_view = angle_of_view(agent_meta, obj_relation_attribute["position"], horizontal_view_angle)
                obj_angle_of_view.append(angle_view)

    return np.array(obj_relations), np.array(obj_relation_triplets), np.array(obj_attributes), np.array(obj_angle_of_view)


def _set_obj_attribute(obj_relation_attribute):
    obj_attribute = [1 if obj_relation_attribute[attr] else 0 for attr in ATTRIBUTE]
    return np.array(obj_attribute)


def _search_boxes_index(boxes_id, parentReceptacles_id):
    for j, object_id in enumerate(boxes_id):
        if parentReceptacles_id == object_id:
            # print("relation\n", parentReceptacles_id)
            return j, RELATION["parentReceptacles"], True
    return 0, 0, False


def angle_of_view(agent_meta, object_position, horizontal_view_angle):
    if agent_meta:
        agent_position = agent_meta["position"]
        x, y, z = object_position['x']-agent_position['x'], object_position['y']-agent_position['y'], object_position['z']-agent_position['z']
        vertical_angle = math.atan2(y, math.hypot(x, z))
        radians = math.radians(horizontal_view_angle)
        horizontal_angle = math.atan2(z, x)
        horizontal_angle = horizontal_angle+radians
        return np.array([horizontal_angle, vertical_angle])
    else:
        return np.array([0, 0])