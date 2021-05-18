import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
from models.utils.metric import AccuracyMetric
from icecream import ic

not_perfect_list = [
    'pick_and_place_simple-SprayBottle-None-Toilet-422/trial_T20190909_124852_071149',
    'pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-14/trial_T20190910_120350_730711',
    'pick_heat_then_place_in_recep-PotatoSliced-None-SinkBasin-23/trial_T20190907_123248_978930',
    'pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-5/trial_T20190908_003714_311231',
    'pick_two_obj_and_place-SoapBar-None-BathtubBasin-422/trial_T20190909_133405_341380',
    'pick_two_obj_and_place-Candle-None-Shelf-422/trial_T20190906_192421_941599',
    # meta
    'pick_heat_then_place_in_recep-Potato-None-Fridge-27/trial_T20190908_143748_027076', 'pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141', 'pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141', 'look_at_obj_in_light-Pillow-None-DeskLamp-319/trial_T20190907_224211_927258', 'pick_cool_then_place_in_recep-PotatoSliced-None-GarbageCan-11/trial_T20190909_013637_168506', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449', 'look_at_obj_in_light-Pillow-None-DeskLamp-319/trial_T20190907_224211_927258', 'pick_and_place_with_movable_recep-Spatula-Pot-DiningTable-26/trial_T20190908_170246_373064', 'pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433', 'pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811', 'look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182531_510491', 'look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182720_056041', 'pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539', 'pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449', 'pick_and_place_with_movable_recep-Spatula-Pot-DiningTable-26/trial_T20190908_170246_373064', 'look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182720_056041', 'pick_heat_then_place_in_recep-Potato-None-Fridge-27/trial_T20190908_143748_027076', 'pick_clean_then_place_in_recep-Cloth-None-Toilet-415/trial_T20190909_022341_367862', 'pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614', 'look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182531_510491', 'pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388', 'pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-5/trial_T20190908_003743_285560', 'pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614', 'look_at_obj_in_light-Laptop-None-DeskLamp-309/trial_T20190907_185744_816357', 'look_at_obj_in_light-Laptop-None-DeskLamp-309/trial_T20190907_185744_816357', 'pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630', 'pick_clean_then_place_in_recep-Spoon-None-DiningTable-20/trial_T20190909_034621_566395', 'pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167', 'look_at_obj_in_light-Laptop-None-DeskLamp-309/trial_T20190907_185728_635748', 'pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388', 'pick_cool_then_place_in_recep-PotatoSliced-None-GarbageCan-11/trial_T20190909_013637_168506', 'pick_and_place_with_movable_recep-Spatula-Pot-DiningTable-26/trial_T20190908_170246_373064', 'pick_and_place_simple-Tomato-None-DiningTable-26/trial_T20190908_010933_200567', 'pick_clean_then_place_in_recep-Cloth-None-Toilet-415/trial_T20190909_022341_367862', 'pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539', 'pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167', 'pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879', 'pick_and_place_with_movable_recep-KeyChain-Plate-Shelf-214/trial_T20190908_221257_299998', 'pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879', 'pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630', 'pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141', 'pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433', 'look_at_obj_in_light-Pillow-None-DeskLamp-319/trial_T20190907_224211_927258', 'pick_cool_then_place_in_recep-PotatoSliced-None-GarbageCan-11/trial_T20190909_013637_168506', 'look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182720_056041', 'pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173', 'pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173', 'pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572', 'pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449', 'pick_heat_then_place_in_recep-Potato-None-Fridge-27/trial_T20190908_143748_027076', 'look_at_obj_in_light-Laptop-None-DeskLamp-309/trial_T20190907_185728_635748', 'pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-5/trial_T20190908_003743_285560', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449', 'pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674', 'look_at_obj_in_light-Laptop-None-DeskLamp-319/trial_T20190908_182531_510491', 'pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614', 'pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811', 'pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630', 'pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433', 'pick_clean_then_place_in_recep-Spoon-None-DiningTable-20/trial_T20190909_034621_566395', 'pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572', 'pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449', 'pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614', 'pick_and_place_simple-Tomato-None-DiningTable-26/trial_T20190908_010933_200567', 'pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572', 'pick_and_place_simple-Tomato-None-DiningTable-26/trial_T20190908_010933_200567', 'pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449', 'pick_and_place_with_movable_recep-KeyChain-Plate-Shelf-214/trial_T20190908_221257_299998', 'pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811', 'pick_clean_then_place_in_recep-Cloth-None-Toilet-415/trial_T20190909_022341_367862', 'pick_and_place_with_movable_recep-KeyChain-Plate-Shelf-214/trial_T20190908_221257_299998', 'pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167', 'pick_clean_then_place_in_recep-Spoon-None-DiningTable-20/trial_T20190909_034621_566395', 'look_at_obj_in_light-Laptop-None-DeskLamp-309/trial_T20190907_185728_635748', 'look_at_obj_in_light-Laptop-None-DeskLamp-309/trial_T20190907_185744_816357', 'pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674', 'pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173', 'pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674', 'pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-5/trial_T20190908_003743_285560',
    # third partyframe feat
    'pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210805_698141', 'pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-26/trial_T20190908_162237_908840', 'pick_and_place_with_movable_recep-Spoon-Cup-SinkBasin-23/trial_T20190908_190304_977742', 'pick_heat_then_place_in_recep-Potato-None-Fridge-27/trial_T20190908_143748_027076', 'pick_cool_then_place_in_recep-PotatoSliced-None-GarbageCan-11/trial_T20190909_013637_168506', 'pick_and_place_with_movable_recep-AppleSliced-Bowl-Fridge-21/trial_T20190908_054316_003433', 'pick_cool_then_place_in_recep-AppleSliced-None-DiningTable-27/trial_T20190907_171803_405680', 'pick_cool_then_place_in_recep-Pan-None-StoveBurner-23/trial_T20190906_215826_707811', 'pick_cool_then_place_in_recep-LettuceSliced-None-SinkBasin-4/trial_T20190909_101847_813539', 'pick_clean_then_place_in_recep-LettuceSliced-None-Fridge-11/trial_T20190918_174139_904388', 'pick_and_place_with_movable_recep-Pen-Bowl-Dresser-311/trial_T20190908_170820_174380', 'pick_heat_then_place_in_recep-Egg-None-Fridge-13/trial_T20190907_151643_465634', 'pick_cool_then_place_in_recep-PotatoSliced-None-CounterTop-19/trial_T20190909_053101_102010', 'pick_and_place_with_movable_recep-Ladle-Bowl-SinkBasin-30/trial_T20190907_143416_683614', 'pick_cool_then_place_in_recep-Lettuce-None-SinkBasin-23/trial_T20190908_173530_026785', 'pick_and_place_with_movable_recep-Spatula-Pan-DiningTable-28/trial_T20190907_222606_903630', 'pick_and_place_simple-ToiletPaper-None-ToiletPaperHanger-407/trial_T20190909_081822_309167', 'pick_clean_then_place_in_recep-Potato-None-DiningTable-4/trial_T20190909_113933_070196', 'pick_and_place_simple-Tomato-None-DiningTable-26/trial_T20190908_010933_200567', 'pick_and_place_with_movable_recep-Spoon-Bowl-SinkBasin-27/trial_T20190907_213616_713879', 'look_at_obj_in_light-Pen-None-DeskLamp-316/trial_T20190908_061814_700195', 'pick_and_place_with_movable_recep-LettuceSliced-Pan-DiningTable-28/trial_T20190906_232604_097173', 'pick_and_place_with_movable_recep-LettuceSliced-Pot-DiningTable-21/trial_T20190907_160923_689765', 'pick_cool_then_place_in_recep-Plate-None-Shelf-20/trial_T20190907_034714_802572', 'pick_clean_then_place_in_recep-Ladle-None-Drawer-4/trial_T20190909_161523_929674', 'pick_two_obj_and_place-LettuceSliced-None-Fridge-1/trial_T20190906_181627_504449',
    # instance
    'pick_and_place_simple-BaseballBat-None-Bed-322/trial_T20190908_115151_138786', 'pick_cool_then_place_in_recep-BreadSliced-None-GarbageCan-19/trial_T20190908_174425_766449', 'pick_two_obj_and_place-SoapBar-None-Drawer-421/trial_T20190908_130615_278829', 'pick_heat_then_place_in_recep-TomatoSliced-None-Fridge-23/trial_T20190909_023439_405422', 'pick_two_obj_and_place-SoapBar-None-Drawer-421/trial_T20190908_130615_278829', 'pick_clean_then_place_in_recep-Lettuce-None-DiningTable-20/trial_T20190906_191212_598810', 'pick_heat_then_place_in_recep-Tomato-None-Fridge-25/trial_T20190909_003326_516119', 'pick_two_obj_and_place-SoapBar-None-Drawer-421/trial_T20190908_130732_924172', 'pick_two_obj_and_place-SoapBar-None-Drawer-421/trial_T20190908_130732_924172', 'look_at_obj_in_light-Plate-None-FloorLamp-211/trial_T20190907_131111_340904', 'pick_cool_then_place_in_recep-Mug-None-SinkBasin-6/trial_T20190907_125559_972660', 'pick_two_obj_and_place-CD-None-Drawer-306/trial_T20190907_025539_829173', 'pick_heat_then_place_in_recep-TomatoSliced-None-Fridge-23/trial_T20190909_023439_405422', 'look_at_obj_in_light-Plate-None-FloorLamp-211/trial_T20190907_131111_340904', 'pick_and_place_simple-BaseballBat-None-Bed-322/trial_T20190908_115151_138786', 'pick_two_obj_and_place-SoapBar-None-Drawer-421/trial_T20190908_130615_278829', 'pick_two_obj_and_place-CellPhone-None-Desk-312/trial_T20190908_122308_202330', 'pick_and_place_simple-BaseballBat-None-Bed-322/trial_T20190908_115151_138786', 'pick_and_place_simple-BaseballBat-None-Bed-322/trial_T20190908_115151_138786', 'pick_heat_then_place_in_recep-Tomato-None-Fridge-25/trial_T20190909_003326_516119', 'pick_heat_then_place_in_recep-Tomato-None-Fridge-25/trial_T20190909_003326_516119', 'pick_two_obj_and_place-CD-None-Drawer-306/trial_T20190907_025539_829173', 'pick_two_obj_and_place-SoapBar-None-Drawer-421/trial_T20190908_130732_924172', 'pick_two_obj_and_place-CellPhone-None-Desk-312/trial_T20190908_122308_202330', 'pick_cool_then_place_in_recep-Mug-None-SinkBasin-6/trial_T20190907_125559_972660', 'pick_cool_then_place_in_recep-Mug-None-SinkBasin-6/trial_T20190907_125559_972660', 'pick_cool_then_place_in_recep-BreadSliced-None-GarbageCan-19/trial_T20190908_174425_766449', 'pick_cool_then_place_in_recep-BreadSliced-None-GarbageCan-19/trial_T20190908_174425_766449', 'pick_clean_then_place_in_recep-Lettuce-None-DiningTable-20/trial_T20190906_191212_598810', 'look_at_obj_in_light-Plate-None-FloorLamp-211/trial_T20190907_131111_340904', 'pick_two_obj_and_place-CellPhone-None-Desk-312/trial_T20190908_122308_202330', 'pick_clean_then_place_in_recep-Lettuce-None-DiningTable-20/trial_T20190906_191212_598810', 'pick_heat_then_place_in_recep-TomatoSliced-None-Fridge-23/trial_T20190909_023439_405422', 'pick_two_obj_and_place-CD-None-Drawer-306/trial_T20190907_025539_829173',
    # sgg_instance
    'pick_clean_then_place_in_recep-Cup-None-Microwave-1/_150843_298006', 'pick_clean_then_place_in_recep-Cup-N/trial_T20190909_150843_298006', 'pick_clean_then_placNone-Microwave-1/trial_T20190909_150843_298006'
]


class Module(nn.Module):

    def __init__(self, args, vocab):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.config = args.config_file
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

    def run_train(self, splits, args=None, optimizer=None):
        '''
        training loop
        '''
        # args
        args = args or self.args

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

        # splits
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        train = [t for t in train for task_type in task_types if task_type in t['task']]
        valid_seen = [t for t in valid_seen for task_type in task_types if task_type in t['task']]
        valid_unseen = [t for t in valid_unseen for task_type in task_types if task_type in t['task']]

        train = [t for t in train if not t['task'] in not_perfect_list]
        valid_seen = [t for t in valid_seen if not t['task'] in not_perfect_list]
        valid_unseen = [t for t in valid_unseen if not t['task'] in not_perfect_list]

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:5]
            valid_seen = valid_seen[:5]
            valid_unseen = valid_unseen[:5]

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)
        self.accuracy_metric = AccuracyMetric()

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)

        # display dout
        print("Saving to: %s" % self.args.dout)
        import time
        start_time = time.time()
        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.train()
            self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            # p_train = {}
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            print("train")
            for batch, feat in self.iterate(train, args.batch):
                pass

            # compute metrics for valid_seen
            print("m_valid_seen")
            self.run_pred(valid_seen, args=args, name='valid_seen', iter=0)

            # compute metrics for valid_unseen
            print("m_valid_unseen")
            self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=0)
            print("--- %s seconds ---" % (time.time() - start_time))
            print(self.missing)
            ic(args.dout)
            return

    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        with torch.no_grad():
            for batch, feat in self.iterate(dev, args.batch):
                pass

    def finish_of_episode(self):
        raise NotImplementedError()

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['ann']['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        # import pdb; pdb.set_trace()
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            tasks = data[i:i+batch_size]
            batch = [self.load_task_json(task) for task in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True