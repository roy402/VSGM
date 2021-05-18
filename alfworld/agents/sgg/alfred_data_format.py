import os
import sys
import numpy as np
import torch
import random
import json
from PIL import Image
from torch.utils.data import Dataset
sys.path.insert(0, os.environ['ALFWORLD_ROOT'])
sys.path.insert(0, os.path.join(os.environ['ALFWORLD_ROOT'], 'agents'))
sys.path.insert(0, os.environ["GRAPH_RCNN_ROOT"])
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
import sgg.parser_scene as parser_scene
from icecream import ic


def rgb():
    SAVE_PATH = os.path.join(os.getcwd(), "semantic_graph/rgb_feature")
    OBJECT_RGB_FEATURE = "object_rgb_feature.json"
    return SAVE_PATH, OBJECT_RGB_FEATURE


def mask():
    SAVE_PATH = os.path.join(os.getcwd(), "semantic_graph/mask_feature")
    OBJECT_RGB_FEATURE = "object_mask_feature.json"
    return SAVE_PATH, OBJECT_RGB_FEATURE


def sgg_mask():
    SAVE_PATH = os.path.join(os.getcwd(), "semantic_graph/sgg_mask_feature")
    OBJECT_RGB_FEATURE = "object_sgg_mask_feature.json"
    return SAVE_PATH, OBJECT_RGB_FEATURE

def sgg_mask_2048():
    SAVE_PATH = os.path.join(os.getcwd(), "semantic_graph/sgg_mask_feature_2048")
    OBJECT_RGB_FEATURE = "object_sgg_mask_feature_2048.json"
    return SAVE_PATH, OBJECT_RGB_FEATURE


SAVE_PATH, OBJECT_RGB_FEATURE = rgb()
SAVE_PATH, OBJECT_RGB_FEATURE = mask()
SAVE_PATH, OBJECT_RGB_FEATURE = sgg_mask()
SAVE_PATH, OBJECT_RGB_FEATURE = sgg_mask_2048()
OBJECT_ATTRIBUTE = "object_attribute.json"


# Transform
class TransMetaData():

    def __init__(self, cfg, FOR_SGG_TRAIN_LABEL=False):
        super(TransMetaData, self).__init__()
        self.cfg = cfg
        self.transforms = parser_scene.get_transform(cfg, train=True)
        '''
        graph-rcnn need
        '''
        # ['AlarmClock', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'Cart', 'CellPhone', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Microwave', 'Mug', 'Newspaper', 'Ottoman', 'PaintingHanger', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveBurner', 'StoveKnob', 'TVStand', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'TowelHolder', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
        self.SGG_train_object_classes = parser_scene.get_object_classes(cfg.ALFREDTEST.object_types)
        # {'__background__': 0, 'AlarmClock': 1, 'Apple': 2, 'AppleSliced': 3, 'ArmChair': 4, 'BaseballBat': 5, 'BasketBall': 6, 'Bathtub': 7, 'BathtubBasin': 8, 'Bed': 9, 'Book': 10, 'Bowl': 11, 'Box': 12, 'Bread': 13, 'BreadSliced': 14, 'ButterKnife': 15, 'CD': 16, 'Cabinet': 17, 'Candle': 18, 'Cart': 19, 'CellPhone': 20, 'Cloth': 21, 'CoffeeMachine': 22, 'CoffeeTable': 23, 'CounterTop': 24, 'CreditCard': 25, 'Cup': 26, 'Desk': 27, 'DeskLamp': 28, 'DiningTable': 29, 'DishSponge': 30, 'Drawer': 31, 'Dresser': 32, 'Egg': 33, 'Faucet': 34, 'FloorLamp': 35, 'Fork': 36, 'Fridge': 37, 'GarbageCan': 38, 'Glassbottle': 39, 'HandTowel': 40, 'HandTowelHolder': 41, 'HousePlant': 42, 'Kettle': 43, 'KeyChain': 44, 'Knife': 45, 'Ladle': 46, 'Laptop': 47, 'LaundryHamper': 48, 'LaundryHamperLid': 49, 'Lettuce': 50, 'LettuceSliced': 51, 'LightSwitch': 52, 'Microwave': 53, 'Mug': 54, 'Newspaper': 55, 'Ottoman': 56, 'PaintingHanger': 57, 'Pan': 58, 'PaperTowel': 59, 'PaperTowelRoll': 60, 'Pen': 61, 'Pencil': 62, 'PepperShaker': 63, 'Pillow': 64, 'Plate': 65, 'Plunger': 66, 'Pot': 67, 'Potato': 68, 'PotatoSliced': 69, 'RemoteControl': 70, 'Safe': 71, 'SaltShaker': 72, 'ScrubBrush': 73, 'Shelf': 74, 'ShowerDoor': 75, 'SideTable': 76, 'Sink': 77, 'SinkBasin': 78, 'SoapBar': 79, 'SoapBottle': 80, 'Sofa': 81, 'Spatula': 82, 'Spoon': 83, 'SprayBottle': 84, 'Statue': 85, 'StoveBurner': 86, 'StoveKnob': 87, 'TVStand': 88, 'TeddyBear': 89, 'Television': 90, 'TennisRacket': 91, 'TissueBox': 92, 'Toaster': 93, 'Toilet': 94, 'ToiletPaper': 95, 'ToiletPaperHanger': 96, 'ToiletPaperRoll': 97, 'Tomato': 98, 'TomatoSliced': 99, 'Towel': 100, 'TowelHolder': 101, 'Vase': 102, 'Watch': 103, 'WateringCan': 104, 'WineBottle': 105}
        object_classes = ['__background__'] + self.SGG_train_object_classes
        self.class_to_ind = parser_scene.get_dict_class_to_ind(object_classes)
        # ['__background__', 'AlarmClock', 'Apple', 'AppleSliced', 'ArmChair', 'BaseballBat', 'BasketBall', 'Bathtub', 'BathtubBasin', 'Bed', 'Book', 'Bowl', 'Box', 'Bread', 'BreadSliced', 'ButterKnife', 'CD', 'Cabinet', 'Candle', 'Cart', 'CellPhone', 'Cloth', 'CoffeeMachine', 'CoffeeTable', 'CounterTop', 'CreditCard', 'Cup', 'Desk', 'DeskLamp', 'DiningTable', 'DishSponge', 'Drawer', 'Dresser', 'Egg', 'Faucet', 'FloorLamp', 'Fork', 'Fridge', 'GarbageCan', 'Glassbottle', 'HandTowel', 'HandTowelHolder', 'HousePlant', 'Kettle', 'KeyChain', 'Knife', 'Ladle', 'Laptop', 'LaundryHamper', 'LaundryHamperLid', 'Lettuce', 'LettuceSliced', 'LightSwitch', 'Microwave', 'Mug', 'Newspaper', 'Ottoman', 'PaintingHanger', 'Pan', 'PaperTowel', 'PaperTowelRoll', 'Pen', 'Pencil', 'PepperShaker', 'Pillow', 'Plate', 'Plunger', 'Pot', 'Potato', 'PotatoSliced', 'RemoteControl', 'Safe', 'SaltShaker', 'ScrubBrush', 'Shelf', 'ShowerDoor', 'SideTable', 'Sink', 'SinkBasin', 'SoapBar', 'SoapBottle', 'Sofa', 'Spatula', 'Spoon', 'SprayBottle', 'Statue', 'StoveBurner', 'StoveKnob', 'TVStand', 'TeddyBear', 'Television', 'TennisRacket', 'TissueBox', 'Toaster', 'Toilet', 'ToiletPaper', 'ToiletPaperHanger', 'ToiletPaperRoll', 'Tomato', 'TomatoSliced', 'Towel', 'TowelHolder', 'Vase', 'Watch', 'WateringCan', 'WineBottle']
        self.SGG_result_ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                                                self.class_to_ind[k])
        # cfg.ind_to_class = self.SGG_result_ind_to_classes
        self.predicate_to_ind = parser_scene.get_dict_predicate_to_ind()
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                        self.predicate_to_ind[k])
        self.FOR_SGG_TRAIN_LABEL = FOR_SGG_TRAIN_LABEL
        ic(len(self.SGG_train_object_classes))
        ic(len(self.SGG_result_ind_to_classes))
        ic(self.ind_to_predicates)

    def trans_object_meta_data_to_relation_and_attribute(self, data_obj_relation_attribute, boxes_id=None, boxes_labels=None, agent_meta=None, horizontal_view_angle=0):
        if boxes_id is None:
            boxes_id, boxes_labels = [], []
            for obj_relation_attribute in data_obj_relation_attribute:
                if obj_relation_attribute["visible"] == True and obj_relation_attribute["objectType"] in self.SGG_result_ind_to_classes:
                    boxes_id.append(obj_relation_attribute["objectId"])
                    class_idx = self.SGG_result_ind_to_classes.index(obj_relation_attribute["objectType"])
                    boxes_labels.append(class_idx)
        obj_relations, obj_relation_triplets, obj_attribute, obj_angle_of_view = parser_scene.transfer_object_meta_data_to_relation_and_attribute(
            boxes_id, data_obj_relation_attribute, horizontal_view_angle, agent_meta)
        '''
         'pred_labels': array([[0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 1., 0., 0., 1.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.],
                               [1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 0.]]), 
         'relation_labels': array([[6, 0, 1],
                                   [2, 1, 1],
                                   [2, 4, 1],
                                   [2, 7, 1]]), 
         'attributes': array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                               0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                               0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0],
                              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                               0]]), 
         'objectIds': ['Spatula|+03.07|+00.76|-00.03', 'ButterKnife|+02.97|+00.90|-00.70', 'CounterTop|+03.11|+00.94|+00.02', 'Sink|+03.08|+00.89|+00.09', 'HousePlant|+03.20|+00.89|-00.37', 'Window|+03.70|+01.68|+00.05', 'Sink|+03.08|+00.89|+00.09|SinkBasin', 'Faucet|+03.31|+00.89|+00.09'], 'labels': array(['Spatula', 'ButterKnife', 'CounterTop', 'Sink', 'HousePlant',
                       'Window', 'SinkBasin', 'Faucet'], dtype='<U11')}
        '''

        dict_obj_rel_attr = {
            "pred_labels": obj_relations,
            "relation_labels": torch.from_numpy(obj_relation_triplets).to(dtype=torch.float),
            "attributes": torch.from_numpy(obj_attribute).to(dtype=torch.float),
            "angle_of_views": torch.from_numpy(obj_angle_of_view).to(dtype=torch.float),
            "objectIds": boxes_id,
            "labels": torch.tensor(boxes_labels).to(dtype=torch.float),
        }
        return dict_obj_rel_attr

    def trans_meta_data_to_sgg(self, mask, color_to_object, data_obj_relation_attribute):
        object_classes = self.SGG_train_object_classes if self.FOR_SGG_TRAIN_LABEL else self.SGG_result_ind_to_classes
        masks, boxes, boxes_labels, boxes_id = parser_scene.transfer_mask_semantic_to_bbox_label(
            mask,
            color_to_object,
            object_classes,
            data_obj_relation_attribute,
        )
        dict_obj_rel_attr = self.trans_object_meta_data_to_relation_and_attribute(
            data_obj_relation_attribute,
            boxes_id,
            boxes_labels
        )
        sgg_data = dict_obj_rel_attr
        sgg_data["boxes"] = boxes
        sgg_data["objectIds"] = boxes_id
        return sgg_data


'''
data from gen/scripts/augment_sgg_trajectories.py
'''


class AlfredDataset(Dataset):
    def __init__(self, cfg, FOR_SGG_TRAIN_LABEL=False):
        from sys import platform
        if platform == "win32":
            cfg.ALFREDTEST.data_path = "D:\\alfred\\alfworld\\detector\\data\\test"
        self.root = cfg.ALFREDTEST.data_path
        self.cfg = cfg

        ''''''
        # load all image files, sorting them to
        # ensure that they are aligned
        self.get_data_files(self.root, balance_scenes=cfg.ALFREDTEST.balance_scenes)

        self.trans_meta_data = TransMetaData(cfg, FOR_SGG_TRAIN_LABEL)
        # train
        self.SGG_train_object_classes = self.trans_meta_data.SGG_train_object_classes
        self.ind_to_classes = self.SGG_train_object_classes

        self.class_to_ind = self.trans_meta_data.class_to_ind
        self.SGG_result_ind_to_classes = self.trans_meta_data.SGG_result_ind_to_classes
        self.predicate_to_ind = self.trans_meta_data.predicate_to_ind
        self.ind_to_predicates = self.trans_meta_data.ind_to_predicates

    def get_data_files(self, root, balance_scenes=False):
        if balance_scenes:
            kitchen_path = os.path.join(root, 'kitchen', 'images')
            living_path = os.path.join(root, 'living', 'images')
            bedroom_path = os.path.join(root, 'bedroom', 'images')
            bathroom_path = os.path.join(root, 'bathroom', 'images')

            kitchen = list(sorted(os.listdir(kitchen_path)))
            living = list(sorted(os.listdir(living_path)))
            bedroom = list(sorted(os.listdir(bedroom_path)))
            bathroom = list(sorted(os.listdir(bathroom_path)))

            min_size = min(len(kitchen), len(living), len(bedroom), len(bathroom))
            kitchen = [os.path.join(kitchen_path, f) for f in random.sample(
                kitchen, int(min_size*self.cfg.ALFREDTEST.kitchen_factor))]
            living = [os.path.join(living_path, f) for f in random.sample(
                living, int(min_size*self.cfg.ALFREDTEST.living_factor))]
            bedroom = [os.path.join(bedroom_path, f) for f in random.sample(
                bedroom, int(min_size*self.cfg.ALFREDTEST.bedroom_factor))]
            bathroom = [os.path.join(bathroom_path, f) for f in random.sample(
                bathroom, int(min_size*self.cfg.ALFREDTEST.bathroom_factor))]

            self.imgs = kitchen + living + bedroom + bathroom
            self.masks = [f.replace("images", "masks") for f in self.imgs]
            self.metas = [f.replace("images", "meta").replace(".png", ".json") for f in self.imgs]
            self.sgg_metas = [f.replace("images", "sgg_meta").replace(".png", ".json")
                              for f in self.imgs]
        else:
            self.imgs = [os.path.join(root, "images", f)
                         for f in list(sorted(os.listdir(os.path.join(root, "images"))))]
            self.masks = [os.path.join(root, "masks", f)
                          for f in list(sorted(os.listdir(os.path.join(root, "masks"))))]
            self.metas = [os.path.join(root, "meta", f)
                          for f in list(sorted(os.listdir(os.path.join(root, "meta"))))]
            self.sgg_metas = [os.path.join(root, "sgg_meta", f)
                              for f in list(sorted(os.listdir(os.path.join(root, "sgg_meta"))))]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load images ad masks
        # print("care for img_path = self.masks[idx]")
        img_path = self.masks[idx]
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        meta_path = self.metas[idx]
        sgg_meta_path = self.sgg_metas[idx]

        # print("Opening: %s" % (self.imgs[idx]))

        with open(meta_path, 'r') as f:
            color_to_object = json.load(f)
        with open(sgg_meta_path, 'r') as f:
            data_obj_relation_attribute = json.load(f)

        img = Image.open(img_path)#.convert("RGB")
        # note that we haven't converted the mask to RGB,
        #
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)

        sgg_data = self.trans_meta_data.trans_meta_data_to_sgg(
            mask,
            color_to_object,
            data_obj_relation_attribute
        )
        boxes = sgg_data["boxes"]
        labels = sgg_data["labels"]
        obj_relations = sgg_data["pred_labels"]
        relation_labels = sgg_data["relation_labels"]
        attributes = sgg_data["attributes"]
        objectIds = sgg_data["objectIds"]
        angle_of_views = sgg_data["angle_of_views"]

        if len(boxes) == 0:
            return None, None

        width, height = img.size
        target_raw = BoxList(boxes, (width, height), mode="xyxy")
        # img, target = img, target_raw
        img, target = self.trans_meta_data.transforms(img, target_raw)
        target.add_field("labels", labels)
        target.add_field("pred_labels", torch.from_numpy(obj_relations).to(dtype=torch.float))
        target.add_field("relation_labels", relation_labels)
        target.add_field("attributes", attributes)
        target.add_field("objectIds", objectIds)
        target.add_field("angle_of_views", angle_of_views)
        target = target.clip_to_image(remove_empty=False)

        # sgg graph training
        return img, target, idx, Image.open(self.imgs[idx]).convert("RGB")
        return img, target, idx

    def get_img_info(self, img_id):
        # w, h = self.im_sizes[img_id, :]
        return {"height": self.cfg.ALFREDTEST.height, "width": self.cfg.ALFREDTEST.weight}

    def map_class_id_to_class_name(self, class_id):
        return self.trans_meta_data.SGG_train_object_classes[class_id]

    def get_PIL_img(self, idx):
        img_path = self.imgs[idx]
        img_path = self.masks[idx]
        return Image.open(img_path).convert('RGB')

def main(cfgs):
    from random import shuffle
    import cv2
    from torchvision.transforms import functional as F
    import time
    cfg = cfgs['semantic_cfg']
    alfred_dataset = AlfredDataset(cfg, FOR_SGG_TRAIN_LABEL=False)
    if True:
        test(alfred_dataset)
        return
    if 'sgg_cfg' in cfgs:
        extractor = get_sgg_model(cfgs, alfred_dataset.trans_meta_data)
    else:
        extractor = get_resnet_model(cfg)
    object_vision_feature = {
        "0": [0]*512,
        # ...
    }
    object_attribute = {
        "0": [0]*23,
        # ...
    }
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    if os.path.isfile(os.path.join(SAVE_PATH, OBJECT_RGB_FEATURE)):
        with open(os.path.join(SAVE_PATH, OBJECT_RGB_FEATURE), "r") as f:
            object_vision_feature = json.load(f)

    object_not_save = [i for i in range(len(alfred_dataset.SGG_result_ind_to_classes))]
    always_not_found = [0, 3, 14, 51, 57, 59, 69, 97, 99]
    # sgg_always_not_found = [0, 1, 3, 14, 51, 57, 59, 69, 97, 99]
    print(alfred_dataset.SGG_result_ind_to_classes)
    print("Always not found: ", [alfred_dataset.SGG_result_ind_to_classes[i] for i in always_not_found])

    def gen_object_rgb_feature():
        indices = list(range(len(alfred_dataset)))
        shuffle(indices)
        for i in indices:
            img, target, idx = alfred_dataset[i]
            dict_target = {
                "labels": target.extra_fields["labels"],
                "obj_relations": target.extra_fields["pred_labels"],
                "relation_labels": target.extra_fields["relation_labels"],
                "attributes": target.extra_fields["attributes"],
                "objectIds": target.extra_fields["objectIds"],
            }
            img = F.to_pil_image(img)
            for ind, label in enumerate(target.extra_fields["labels"].numpy().astype(int)):
                if label in object_not_save:
                    bbox = target.bbox[ind].numpy().astype(int)
                    crop_img = img.crop(bbox)
                    if crop_img.size[0] * crop_img.size[1] < 3000:# or min(bbox) == 0 or max(bbox) == 400:
                        continue
                    object_not_save.remove(label)
                    print("remove: ", label)
                    print("bbox:", bbox)
                    print("object: ", alfred_dataset.SGG_result_ind_to_classes[label])
                    print("objectIds: ", target.extra_fields["objectIds"][ind])
                    rgb_feature = extractor.featurize([crop_img], batch=1)
                    if len(rgb_feature.shape) >= 3:
                        GAP_rgb_feature = rgb_feature[0].mean([1, 2]).reshape(-1)
                    else:
                        GAP_rgb_feature = rgb_feature.reshape(-1)
                    object_vision_feature[str(label)] = GAP_rgb_feature.to('cpu').numpy().tolist()
                    print(GAP_rgb_feature.shape)
                    crop_img.save(os.path.join(SAVE_PATH, alfred_dataset.SGG_result_ind_to_classes[label] + "_{}.jpg".format(label)))
                    time.sleep(1)
                    print("Can't find object class: ", object_not_save)
            # [0, 3, 14, 51, 57, 59, 69, 97, 99]
            if len(object_not_save) <= 9:
                break
            img.close()
            # import pdb; pdb.set_trace()
        print("Can't find object class: ", object_not_save)
        object_vision_feature["0"] = [0]*GAP_rgb_feature.shape[0]
        for object_label in object_not_save:
            object_vision_feature[str(object_label)] = [0]*GAP_rgb_feature.shape[0]
        with open(os.path.join(SAVE_PATH, OBJECT_RGB_FEATURE), "w") as f:
            json.dump(object_vision_feature, f)

    def gen_object_SGG_feature():
        indices = list(range(len(alfred_dataset)))
        shuffle(indices)
        for i in indices:
            img, target, idx, img2 = alfred_dataset[i]
            img = img.unsqueeze(0)
            sgg_results = extractor.predict(img, idx)
            sgg_result = sgg_results[0]
            img = F.to_pil_image(img[0])
            for ind, label in enumerate(sgg_result["labels"].numpy().astype(int)):
                if label in object_not_save:
                    bbox = sgg_result["bbox"][ind].numpy().astype(int)
                    crop_img = img.crop(bbox)
                    if crop_img.size[0] * crop_img.size[1] < 1500:# or min(bbox) == 0 or max(bbox) == 400:
                        continue
                    object_not_save.remove(label)
                    print("remove: ", label)
                    print("bbox:", bbox)
                    print("object: ", alfred_dataset.SGG_result_ind_to_classes[label])
                    feature = sgg_result["features"][ind]
                    feature = feature.reshape(-1)
                    object_vision_feature[str(label)] = feature.to('cpu').numpy().tolist()
                    print(feature.shape)
                    crop_img.save(os.path.join(SAVE_PATH, alfred_dataset.SGG_result_ind_to_classes[label] + "_{}.jpg".format(label)))
                    img.save(os.path.join(SAVE_PATH, alfred_dataset.SGG_result_ind_to_classes[label] + "_{}i.jpg".format(label)))
                    img2.save(os.path.join(SAVE_PATH, alfred_dataset.SGG_result_ind_to_classes[label] + "_{}o.jpg".format(label)))
                    print("Can't find object class: ", object_not_save)
            # [0, 3, 14, 51, 57, 59, 69, 97, 99]
            if len(object_not_save) <= 9:
                break
            img.close()
            # import pdb; pdb.set_trace()
        print("Can't find object class: ", object_not_save)
        object_vision_feature["0"] = [0]*feature.shape[0]
        for object_label in object_not_save:
            object_vision_feature[str(object_label)] = [0]*feature.shape[0]
        with open(os.path.join(SAVE_PATH, OBJECT_RGB_FEATURE), "w") as f:
            json.dump(object_vision_feature, f)

    def gen_object_attr():
        indices = list(range(len(alfred_dataset)))
        shuffle(indices)
        for i in indices:
            img, target, idx = alfred_dataset[i]
            dict_target = {
                "labels": target.extra_fields["labels"],
                "obj_relations": target.extra_fields["pred_labels"],
                "relation_labels": target.extra_fields["relation_labels"],
                "attributes": target.extra_fields["attributes"],
                "objectIds": target.extra_fields["objectIds"],
            }
            print(target.extra_fields["angle_of_views"])
            img = F.to_pil_image(img)
            for ind, label in enumerate(target.extra_fields["labels"].numpy().astype(int)):
                if label in object_not_save:
                    attribute = target.extra_fields["attributes"][ind].numpy().astype(int).tolist()
                    object_not_save.remove(label)
                    object_attribute[str(label)] = attribute
                    ic(attribute)
                    print("remove: ", label)
                    print("object: ", alfred_dataset.SGG_result_ind_to_classes[label])
                    time.sleep(1)
                    print("Can't find object class: ", object_not_save)
            # [0, 3, 14, 51, 57, 59, 69, 97, 99]
            if len(object_not_save) <= 9:
                break
            img.close()
            # import pdb; pdb.set_trace()
        print("Can't find object class: ", object_not_save)
        for object_label in object_not_save:
            object_attribute[str(object_label)] = [0]*23
        with open(os.path.join(SAVE_PATH, OBJECT_ATTRIBUTE), "w") as f:
            json.dump(object_attribute, f)

    # gen_object_attr()
    object_not_save = [i for i in range(len(alfred_dataset.SGG_result_ind_to_classes))]
    # gen_object_rgb_feature()
    gen_object_SGG_feature()


def test(alfred_dataset):
    indices = list(range(len(alfred_dataset)))
    count = 0
    for i in indices:
        img, target, idx, rgb_img = alfred_dataset[i]
        dict_target = {
            "labels": target.extra_fields["labels"],
            "obj_relations": target.extra_fields["pred_labels"],
            "relation_labels": target.extra_fields["relation_labels"],
            "attributes": target.extra_fields["attributes"],
            "objectIds": target.extra_fields["objectIds"],
        }
        import pdb; pdb.set_trace()


def get_resnet_model(cfg):
    sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))
    from nn.resnet import Resnet
    cfg.gpu = True
    extractor = Resnet(cfg, eval=True)
    return extractor


def get_sgg_model(cfg, trans_MetaData):
    # import sgg
    from sgg.sgg import load_pretrained_model
    cfg_sgg = cfg['sgg_cfg']
    detector = load_pretrained_model(
        cfg_sgg,
        trans_MetaData.transforms,
        trans_MetaData.SGG_result_ind_to_classes,
        'cuda'
        )
    return detector


def get_dataset(cfg, transform=None, FOR_SGG_TRAIN_LABEL=False):
    dataset = AlfredDataset(cfg, FOR_SGG_TRAIN_LABEL=FOR_SGG_TRAIN_LABEL)
    return dataset


if __name__ == "__main__":
    import modules.generic as generic
    cfg = generic.load_config()
    main(cfg)
