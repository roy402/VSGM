import os
import sys
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange
from sys import platform
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], '..', 'graph_analysis'))
import fastText_embedding
from torchnlp.word_to_vector import FastText
from model.dgl_pretrain_hete import THETLOWSG
import pdb

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
        self.vocab = vocab

        self.device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() and args.gpu else "cpu")

        # emb modules
        assert args.demb == 300, "demb dim must be same with fastText model 300 dim"
        # self.ft_model = fastText_embedding.load_model()
        self.ft_model = FastText()
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)
        self.gcn = THETLOWSG(args, self.args.HETAttention, args.dgcnout, self.device)

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

        # splits
        train = splits['train']
        # import pdb; pdb.set_trace()
        # print(train)
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=args.dout)

        # dump config
        fconfig = os.path.join(args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=args.lr)
        optimizer = optimizer or torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.9)

        """
        Contrastive Learning Para
        """
        margin, contrastive_loss_wt = self.args.contrastive_margin, self.args.contrastive_loss_wt
        contrastive_train_iter = 0
        # assert args.batch >1, "batch size have to > 1 let contrastive learning train"
        # pretrain graph
        # for i in range(200):
        #     self.gcn.train_nodes(optimizer, self.summary_writer)
        # display dout
        print("Saving to: %s" % self.args.dout)
        best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0
        for epoch in trange(0, args.epoch, desc='epoch'):
            m_train = collections.defaultdict(list)
            self.gcn.train_nodes(optimizer, self.summary_writer)
            self.train()
            lr = self.adjust_lr(optimizer, args.lr, epoch, decay_epoch=args.decay_epoch)
            self.summary_writer.add_scalar('train/lr', lr, epoch)
            """
            Contrastive Learning
            """
            for batch, feat in self.iterate_contrastive_data(train, args.batch_contrast):
                optimizer.zero_grad()
                feature_visal, feature_ins = self.forward_visaul_action_instruction(feat)
                if False:
                    loss_video_instruction_pos, loss_video_instruction_neg = self.visaul_instruction_contrastive(feature_visal, feature_ins)
                    loss_video_video_pos, loss_video_video_neg = self.visaul_visaul_contrastive(feature_visal, feature_ins)
                    # positive_sample loss smaller is better to minimize
                    # negative_sample loss bigger is better
                    # negative_sample need to be minus to maximize
                    total_loss_pos = loss_video_instruction_pos + loss_video_video_pos
                    total_loss_neg = margin + loss_video_instruction_neg + loss_video_video_neg
                    total_loss_neg = torch.clamp(total_loss_neg*contrastive_loss_wt, min=0.0)
                    self.summary_writer.add_scalar('contrastive/loss_video_instruction_pos', loss_video_instruction_pos.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/loss_video_instruction_neg', loss_video_instruction_neg.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/loss_video_video_pos', loss_video_video_pos.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/loss_video_video_neg', loss_video_video_neg.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/total_loss_neg', total_loss_neg.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/total_loss_pos', total_loss_pos.item(), contrastive_train_iter)
                    total_loss_pos.backward(retain_graph=True)
                    total_loss_neg.backward()
                else:
                    loss_visal = self.compute_SimCLR_loss(feature_visal)
                    loss_ins = self.compute_SimCLR_loss(feature_ins)
                    total_loss = loss_visal + loss_ins
                    self.summary_writer.add_scalar('contrastive/SimCLR_visual', loss_visal.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/SimCLR_ins', loss_ins.item(), contrastive_train_iter)
                    self.summary_writer.add_scalar('contrastive/SimCLR_total', total_loss.item(), contrastive_train_iter)
                    total_loss.backward()
                optimizer.step()
                contrastive_train_iter += 1
            """
            Ori train action predict
            """
            # p_train = {}
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            for batch, feat in self.iterate(train, args.batch, augmentation=True):
                optimizer.zero_grad()
                out = self.forward(feat)
                preds = self.extract_preds(out, batch, feat)
                # p_train.update(preds)
                loss = self.compute_loss(out, batch, feat)
                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                # optimizer backward pass
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch

            ## compute metrics for train (too memory heavy!)
            # m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            # m_train.update(self.compute_metric(p_train, train))
            # m_train['total_loss'] = sum(total_train_loss) / len(total_train_loss)
            # self.summary_writer.add_scalar('train/total_loss', m_train['total_loss'], train_iter)

            # compute metrics for valid_seen
            p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(valid_seen, args=args, name='valid_seen', iter=valid_seen_iter)
            m_valid_seen.update(self.compute_metric(p_valid_seen, valid_seen))
            m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            self.summary_writer.add_scalar('valid_seen/total_loss', m_valid_seen['total_loss'], valid_seen_iter)

            # compute metrics for valid_unseen
            p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(valid_unseen, args=args, name='valid_unseen', iter=valid_unseen_iter)
            m_valid_unseen.update(self.compute_metric(p_valid_unseen, valid_unseen))
            m_valid_unseen['total_loss'] = float(total_valid_unseen_loss)
            self.summary_writer.add_scalar('valid_unseen/total_loss', m_valid_unseen['total_loss'], valid_unseen_iter)

            stats = {'epoch': epoch,
                     'train': sum(total_train_loss)/len(total_train_loss),
                     'valid_seen': m_valid_seen,
                     'valid_unseen': m_valid_unseen}

            # new best valid_seen loss
            if total_valid_seen_loss < best_loss['valid_seen']:
                print('\nFound new best valid_seen!! Saving...')
                fsave = os.path.join(args.dout, 'best_seen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_seen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_seen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_seen, valid_seen), f, indent=2)
                best_loss['valid_seen'] = total_valid_seen_loss

            # new best valid_unseen loss
            if total_valid_unseen_loss < best_loss['valid_unseen']:
                print('Found new best valid_unseen!! Saving...')
                fsave = os.path.join(args.dout, 'best_unseen.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)
                fbest = os.path.join(args.dout, 'best_unseen.json')
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)

                fpred = os.path.join(args.dout, 'valid_unseen.debug.preds.json')
                with open(fpred, 'wt') as f:
                    json.dump(self.make_debug(p_valid_unseen, valid_unseen), f, indent=2)

                best_loss['valid_unseen'] = total_valid_unseen_loss

            # save the latest checkpoint
            if args.save_every_epoch:
                fsave = os.path.join(args.dout, 'net_epoch_%d.pth' % epoch)
            else:
                fsave = os.path.join(args.dout, 'latest.pth')
            torch.save({
                'metric': stats,
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

            ## debug action output json for train
            # fpred = os.path.join(args.dout, 'train.debug.preds.json')
            # with open(fpred, 'wt') as f:
            #     json.dump(self.make_debug(p_train, train), f, indent=2)

            # write stats
            for split in stats.keys():
                if isinstance(stats[split], dict):
                    for k, v in stats[split].items():
                        self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
            pprint.pprint(stats)

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
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
                self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
            sum_loss = sum(loss.values())
            self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

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

    def get_faxtText_embedding(self, text):
        # self.ft_model["test"]
        return self.ft_model[text]

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['repeat_idx']))

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
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        if platform == "win32":
            return os.path.join(self.args.data, ex['split'], ex['root'].split('\\')[-1])
        else:
            return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size, augmentation=False):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            tasks = data[i:i+batch_size]
            batch = [self.load_task_json(task) for task in tasks]
            feat = self.featurize(batch, augmentation=augmentation)
            yield batch, feat

    def iterate_contrastive_data(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        import random
        dict_data = collections.defaultdict(list)
        for i in range(len(data)):
            dict_data[data[i]["task"]].append(data[i])
        list_data = list(dict_data.values())
        list_data = random.choices(list_data, k=len(list_data)//100)
        for i in trange(0, len(list_data)-batch_size, batch_size, desc='batch'):
            tasks = list_data[i:i+batch_size+1]
            batch_tasks = []
            # [[{'repeat_idx': 0, 'task': 'pick_clean_then_place_in_recep-Bowl-None-Shelf-7/trial_T20190908_152810_145685'}, {'repeat_idx': 1, 'task': 'pick_clean_then_place_in_recep-Bowl-None-Shelf-7/trial_T20190908_152810_145685'}, {'repeat_idx': 2, 'task': 'pick_clean_then_place_in_recep-Bowl-None-Shelf-7/trial_T20190908_152810_145685'}], ...]
            for j in range(len(tasks)):
                if j == len(tasks)-1:
                    break
                # [{'repeat_idx': 1, 'task': 'pick_two_obj_and_place-CreditCard-None-Desk-313/trial_T20190909_104526_715846'}, {'repeat_idx': 1, 'task': 'pick_two_obj_and_place-CreditCard-None-Desk-313/trial_T20190909_104526_715846'}]
                positive_sample_task = random.choices(tasks[j], k=2)
                # [{'repeat_idx': 2, 'task': 'pick_two_obj_and_place-Candle-None-Shelf-429/trial_T20190909_075215_332862'}]
                negative_sample_task = random.choices(tasks[j+1], k=1)
                batch_tasks.extend(positive_sample_task)
                batch_tasks.extend(negative_sample_task)
            batch = [self.load_task_json(task) for task in batch_tasks]
            feat = self.featurize(batch, augmentation=True)
            # import pdb; pdb.set_trace()
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
        lr = init_lr * (0.5 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    @classmethod
    def load(cls, fsave, device=None, use_gpu=None):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave, map_location=device)
        save['args'].gpu = use_gpu if use_gpu == False else save['args'].gpu
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        model = model.to(device=device)
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
