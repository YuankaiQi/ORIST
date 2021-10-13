# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os
import random
from io import open
import numpy as np
import math
from apex.normalization import fused_layer_norm
# from apex import amp
from torch.autograd import Variable
from model.r2r import UniterForR2RAux
from torch.optim import Adam, Adamax
from optim import AdamW
import sys
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from data.data import get_gather_index
from r2r_utils import angle_feature,check_param_across_gpus
from utils.const import IMG_DIM # image feature dim
from utils.logger import LOGGER
from utils.misc import set_dropout, set_random_seed
from collections import defaultdict
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)

from torch.nn.parallel import DistributedDataParallel as DDP

class BaseAgent(object):
    ''' Base class for an R2R agent to generate and save trajectories. '''

    def __init__(self, env, results_path):
        self.env = env
        self.results_path = results_path
        # random.seed(1)
        self.results = {}


    def write_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        with open(self.results_path, 'w') as f:
            json.dump(output, f)

    def get_results(self):
        output = [{'instr_id': k, 'trajectory': v} for k, v in self.results.items()]
        return output

    def rollout(self, **args):
        ''' Return a list of dicts containing instr_id:'xx', path:[(viewpointId, heading_rad, elevation_rad)]  '''
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name + "Agent"]

    def test(self, iters=None, **kwargs):
        self.env.reset_epoch(shuffle=(iters is not None))  # If iters is not none, shuffle the env batch
        # self.losses = []
        self.results = {}
        self.phase = 'test'
        # We rely on env showing the entire batch before repeating anything
        looped = False

        if iters is not None:
            # For each time, it will run the first 'iters' iterations. (It was shuffled before)
            for i in range(iters):
                batch_ids = [self.env.ix + j for j in range(self.env.batch_size)]
                # print(batch_ids)
                self.env.ix += self.env.batch_size
                self.env.prepare_batch(batch_ids)
                for traj in self.rollout(**kwargs):
                    self.results[traj['instr_id']] = traj['path']
        else:  # Do a full round
            while True:
                # batch_ids = [self.env.ix + j for j in range(self.env.batch_size)]
                # batch_ids= [768]
                # self.env.ix += self.env.batch_size
                # self.env.prepare_batch(batch_ids)
                # print(batch_ids)
                for traj in self.rollout(**kwargs):
                    if traj['instr_id'] in self.results:
                        looped = True
                    else:
                        self.results[traj['instr_id']] = traj['path']
                if looped:
                    break


class BertAgent(BaseAgent):
    '''An agent based on vilbert'''
    # For now, the agent can't pick which forward move to make - just the one in the middle
    env_actions = {
        'left': (0, -1, 0),  # left
        'right': (0, 1, 0),  # right
        'up': (0, 0, 1),  # up
        'down': (0, 0, -1),  # down
        'forward': (1, 0, 0),  # forward
        '<end>': (0, 0, 0),  # <end>
        '<start>': (0, 0, 0),  # <start>
        '<ignore>': (0, 0, 0)  # <ignore>
    }

    def __init__(self, env, results_path, tok, episode_len=20, args=None):
        super(BertAgent, self).__init__(env, results_path)
        self.tok = tok
        self.episode_len = episode_len
        self.feature_size = self.env.feature_size if self.env is not None else 0
        self.args = args
        self.cache_candidate_region_id = {}
        self.device = torch.device(f'cuda:{args.local_rank}')


        self.model = UniterForR2RAux.from_pretrained(
            args.model_config, state_dict={}, img_dim=IMG_DIM, tok=tok, args=args)

        # load speaker
        if args.speaker_loss:
            assert args.speaker is not None
            self.fill_in_pretrained_speaker(args.speaker)


        exclude_params = ['r2r_output', 'critic', 'orien_embeddings', 'lstm', 'progress_pred',
                          'angle_pred', 'next_region_pred', 'target_region_pred',
                          'region_pred', 'speaker_decoder']

        set_dropout(self.model, args.dropout, exclude_params=exclude_params)
        self.model.to(self.device)

        # DDP
        self.model = DDP(self.model, device_ids=[args.local_rank], find_unused_parameters=True)

        self.optimizer = self.build_optimizer(args)  # check param whether on gpu

        self.start_epoch = 0
        self.global_step = 0
        self.resumed = False

        if args.resume is not None:#note below code need a workaround to handle optimizer restore
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.local_rank))
            LOGGER.info('load checkpoint %s' % args.resume)

            self.model.load_state_dict(checkpoint["model"])

            if args.resume_optimizer:
                LOGGER.info('resume optimizer')
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                self.global_step = checkpoint["global_step"]
                self.start_epoch = int(checkpoint["epoch_id"]) + 1
                self.best_val = checkpoint["best_val"]
                self.logs = defaultdict(float)
                for k, v in checkpoint["logs"].items():
                    self.logs[k] = v
                self.resumed = True

        else:
            # 'module' should not appear in pretrained_model
            LOGGER.info('Load Model from %s' % args.pretrained_model)
            pm = torch.load(args.pretrained_model,
                                map_location='cuda:{}'.format(args.local_rank))  # map_location='cpu'

            self.fill_in_pretrained_weights(pm)


        self.action_crit = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='sum')
        self.progress_crit = nn.BCELoss() #check here later
        self.angle_crit = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='mean')
        self.speaker_crit = nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        self.region_crit = nn.BCEWithLogitsLoss(reduction='mean')
        self.single_region_crit = nn.CrossEntropyLoss(ignore_index=args.ignoreid, reduction='mean')

        self.median_num_iter = math.ceil(len(self.env.data)/(args.train_batch_size*args.n_gpu)) # iters per gpu
        self.num_train_optimization_steps = args.num_train_epochs*self.median_num_iter # total steps per gpu
        self.warmup_steps = args.warmup_proportion * self.num_train_optimization_steps

        args.warmup_steps = self.warmup_steps
        args.num_train_steps = self.num_train_optimization_steps

        # for orientation target
        self.orien_dict = {
            'forward': 0,
            'backward': 1,
            'left': 2,
            'right': 3
        }

        if self.args.self_train:
            self.drop_env = nn.Dropout(args.speaker_featdropout)

    def fill_in_pretrained_speaker(self, speaker_path):
        state_dict = torch.load(speaker_path)["decoder"]["state_dict"]
        state_keys = list(state_dict.keys())
        for k, p in self.model.speaker_decoder.named_parameters():
            if k in state_dict:
                LOGGER.info('Load %s from pretrained Speaker' % k)
                state_keys.remove(k)
                pretrained_v = state_dict[k].float()
                if pretrained_v.shape == p.data.shape:
                    p.data = pretrained_v
                else:
                    # this is specially for word embedding layers
                    LOGGER.info('%s has different data size with pretrained model, pretrain = %s, cur = %s' % (k, str(pretrained_v.shape), str(p.data.shape)))
                    p.data[:pretrained_v.shape[0], :] = pretrained_v
        if len(state_keys) > 0:
            for temp_key in state_keys:
                LOGGER.info('Not used: %s' % temp_key)

    def fill_in_pretrained_weights(self, state_dict):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if key.startswith('module'):
                new_key = key[7:]
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        state_keys = list(state_dict.keys())
        for nk, p in self.model.named_parameters():
            if nk.startswith('module'):
                k = nk[7:]
            else:
                k = nk
            if k in state_dict:
                LOGGER.info('Load %s from pretrained model' % k)
                state_keys.remove(k)
                pretrained_v = state_dict[k].to(dtype=p.data.dtype)
                if pretrained_v.shape == p.data.shape:
                    p.data = pretrained_v
                else:
                    # this is specially for word embedding layers
                    LOGGER.info('%s has different data size with pretrained model, pretrain = %s, cur = %s' % (k, str(pretrained_v.shape), str(p.data.shape)))
                    p.data[:pretrained_v.shape[0], :] = pretrained_v
        if len(state_keys) > 0:
            for temp_key in state_keys:
                LOGGER.info('Not used: %s' % temp_key)


    def build_optimizer(self, opts):
        """ r2r linear may get larger learning rate """
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        special_params = ['r2r_output', 'critic', 'orien_embeddings', 'lstm', 'progress_pred',
                          'angle_pred', 'next_region_pred', 'target_region_pred',
                          'region_pred', 'speaker_decoder']
        params_body = []
        sp_dict = defaultdict(list)
        for n, p in self.model.named_parameters():
            is_body = True
            for sp in special_params:
                if sp in n:
                    sp_dict[sp].append((n, p))
                    is_body = False
                    break
            if is_body:
                params_body.append((n, p))


        optimizer_grouped_parameters = []
        for key, sp in sp_dict.items():
            decayed_params = {
                        'params': [p for n, p in sp if not any(nd in n for nd in no_decay)],
                        'lr': opts.learning_rate,
                        'weight_decay': opts.weight_decay}
            no_decay_params = {
                        'params': [p for n, p in sp if any(nd in n for nd in no_decay)],
                        'lr': opts.learning_rate,
                        'weight_decay': 0.0}
            optimizer_grouped_parameters.append(decayed_params)
            optimizer_grouped_parameters.append(no_decay_params)

        # body
        body_decay_params = {
            'params': [p for n, p in params_body if not any(nd in n for nd in no_decay)],
            'weight_decay': opts.weight_decay}
        optimizer_grouped_parameters.append(body_decay_params)

        body_no_decay_params = {
            'params': [p for n, p in params_body if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0}
        optimizer_grouped_parameters.append(body_no_decay_params)

        if opts.optim == 'adam':
            OptimCls = Adam
        elif opts.optim == 'adamax':
            OptimCls = Adamax
        elif opts.optim == 'adamw':
            OptimCls = AdamW
        else:
            raise ValueError('invalid optimizer')
        optimizer = OptimCls(optimizer_grouped_parameters,
                                  lr=opts.learning_rate, betas=opts.betas)
        return optimizer

    def _candidate_variable(self, obs):
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candidate_feat = np.zeros((len(obs), max(candidate_leng), self.feature_size + self.args.angle_feat_size),
                                  dtype=np.float32)
        # Note: The candidate_feat at len(ob['candidate']) is the feature for the END
        # which is zero in my implementation
        for i, ob in enumerate(obs):
            for j, c in enumerate(ob['candidate']):
                candidate_feat[i, j, :] = c['feature']  # need to change Image feat to obj features

        return torch.from_numpy(candidate_feat).cuda(), candidate_leng

    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), self.args.views, self.feature_size + self.args.angle_feat_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['feature']   # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def get_input_feat(self, obs):
        input_a_t = np.zeros((len(obs), self.args.angle_feat_size), np.float32)
        for i, ob in enumerate(obs):
            input_a_t[i] = angle_feature(ob['heading'], ob['elevation'])
        input_a_t = torch.from_numpy(input_a_t).cuda()

        f_t = self._feature_variable(obs)  # Image features from obs
        candidate_feat, candidate_leng = self._candidate_variable(obs)

        return input_a_t, f_t, candidate_feat, candidate_leng

    def get_bert_region_feature(self, obs):
        '''return: num_boxes, region_feat, region_loc'''
        candidate_leng = [len(ob['candidate']) + 1 for ob in obs]  # +1 is for the end
        candi_num = sum(candidate_leng)
        mix_num_boxes = []
        mix_boxes_pad = np.zeros((candi_num, self.env.bert_max_region_num, 7), dtype=np.float32)
        mix_features_pad = np.zeros((candi_num, self.env.bert_max_region_num, 2048), dtype=np.float32)
        count = 0
        angle_feat = np.zeros((candi_num,self.args.angle_feat_size),dtype=np.float32)
        candidate_mask = np.zeros((len(obs), max(candidate_leng)), dtype=np.uint8)
        for i, ob in enumerate(obs):
            candidate_mask[i, candidate_leng[i]:] = 1
            for j, c in enumerate(ob['candidate']):
                num_boxes = c['region_num']
                mix_num_boxes.append(num_boxes)
                if self.args.mask_obj and num_boxes > 0:
                    region_feat_shape = c['region_feat'].shape
                    assert num_boxes==region_feat_shape[0]
                    region_mask = np.random.rand(region_feat_shape[0]) < 0.15
                    if num_boxes > 1:
                        # must mask one or more objs if num_boxes > 1
                        while not any(region_mask):
                            region_mask = np.random.rand(region_feat_shape[0]) < 0.15
                    region_mask = np.repeat(np.expand_dims((1-region_mask), axis=1), region_feat_shape[-1], 1)
                    c['region_feat'] = c['region_feat'] * region_mask
                # else:
                mix_features_pad[count,:num_boxes,:] = c['region_feat']
                mix_boxes_pad[count,:num_boxes,:] = c['region_loc']
                angle_feat[count,:]=c['angle_feat']
                count += 1
            # add looking to current vp as end
            num_boxes = self.looking_to_next_vp_feat[i]['region_num']
            mix_num_boxes.append(num_boxes)
            angle_feat[count, : ] = self.looking_to_next_vp_feat[i]['angle_feat']
            mix_features_pad[count, :, :] = self.looking_to_next_vp_feat[i]['region_feat']
            mix_boxes_pad[count, :, :] = self.looking_to_next_vp_feat[i]['region_loc']
            count += 1

        return np.array(mix_num_boxes), mix_features_pad,\
                mix_boxes_pad, angle_feat, \
                candidate_leng,  candidate_mask

    def _teacher_action(self, obs, ended):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:  # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                a[i] = ob['candi_vp_2_id'].get(ob['teacher'], len(ob['candidate']))
        return torch.from_numpy(a).to(self.device)

    def make_equiv_action(self, a_t, perm_obs, perm_idx=None, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """

        def take_action(i, idx, name):
            if type(name) is int:  # Go to the next view
                self.env.env.sims[idx].makeAction(name, 0, 0)
            else:  # Adjust
                self.env.env.sims[idx].makeAction(*self.env_actions[name])
            state = self.env.env.sims[idx].getState()
            if traj is not None:
                traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

        if perm_idx is None:
            perm_idx = range(len(perm_obs))
        for i, idx in enumerate(perm_idx):
            action = a_t[i]
            if action != -1:  # -1 is the <stop> action
                select_candidate = perm_obs[i]['candidate'][action]
                src_point = perm_obs[i]['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point) // 12  # The point idx started from 0
                trg_level = (trg_point) // 12
                while src_level < trg_level:  # Tune up
                    take_action(i, idx, 'up')
                    src_level += 1
                while src_level > trg_level:  # Tune down
                    take_action(i, idx, 'down')
                    src_level -= 1
                while self.env.env.sims[idx].getState().viewIndex != trg_point:  # Turn right until the target
                    take_action(i, idx, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[idx].getState().navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, idx, select_candidate['idx'])

    def process_mini_batchs(self, perm_obs, train_rl, cur_step, speaker_step=False):
        # input_a_t, f_t, candidate_feat, candidate_leng = self.get_input_feat(perm_obs)
        box_num, region_feat, region_loc, candi_angle, candidate_leng,\
                                candidate_mask = self.get_bert_region_feature(perm_obs)
        batch_len = sum(candidate_leng)
        instrs = np.zeros((batch_len, self.env.bert_max_seq_length))
        count = 0
        txt_lens = np.zeros(batch_len,dtype=np.int)

        for i, ob in enumerate(perm_obs):
            candi_num = candidate_leng[i]
            txt_lens[count:count+candi_num] = ob['bert_txt_len']
            instrs[count:count+candi_num, :] = np.tile(np.array(ob['bert_instr_encoding']), (candi_num, 1))
            count += candi_num

        atten_len_max = (txt_lens+box_num).max() + 1 # 1: angle feat
        attn_masks = np.zeros((batch_len,atten_len_max),dtype=np.int64) # indicates where meaningful entities are
        txt_lens = txt_lens.tolist()
        box_num = box_num.tolist()
        for i, (tl, nbb) in enumerate(zip(txt_lens, box_num)):
            attn_masks[i,:tl+nbb+1] = 1 # 1: angle token

        candidate_mask = torch.from_numpy(candidate_mask).to(self.device) # this is prepared to cal loss
        txt_position_ids = torch.arange(0, self.env.bert_max_seq_length, dtype=torch.long
                                    ).unsqueeze(0).to(self.device)
        mini_region_feat = torch.from_numpy(region_feat).to(self.device)

        if self.phase=='train' and self.args.drop_region_feat:
            mini_region_feat = F.dropout(mini_region_feat, p=self.args.region_drop_p)

        mini_region_loc = torch.from_numpy(region_loc).to(self.device)
        mini_orien_feat = torch.from_numpy(candi_angle).to(self.device)
        mini_instr = torch.from_numpy(instrs).long().to(self.device)
        attn_masks = torch.from_numpy(attn_masks).to(self.device)
        gather_index = get_gather_index(txt_lens, box_num, batch_len,
                                        self.env.bert_max_seq_length,
                                        self.env.bert_max_region_num,atten_len_max).to(self.device)

        if speaker_step:
            vln_logit, instra_state, self.state_c0 = self.model(train_rl,
                                                                input_ids=mini_instr,
                                                                position_ids=txt_position_ids,
                                                                img_feat=mini_region_feat,
                                                                img_pos_feat=mini_region_loc,
                                                                orien_feat=mini_orien_feat,
                                                                attn_masks=attn_masks,
                                                                gather_index=gather_index,
                                                                candidate_leng=candidate_leng,
                                                                instr_state=self.last_instr_state,
                                                                c_0=self.state_c0, step=cur_step,
                                                                speaker_step=speaker_step)
            return vln_logit, instra_state, candidate_leng, candidate_mask, box_num, region_feat, region_loc
        else:
            if self.phase == 'train':
                vln_logit, e_reward, instra_state, progress_logits, angle_logits, region_logits, \
                next_region_logits, target_region_logits, self.state_c0 = self.model(train_rl,
                                                                                     input_ids=mini_instr,
                                                                                     position_ids=txt_position_ids,
                                                                                     img_feat=mini_region_feat,
                                                                                     img_pos_feat=mini_region_loc,
                                                                                     orien_feat=mini_orien_feat,
                                                                                     attn_masks=attn_masks,
                                                                                     gather_index=gather_index,
                                                                                     candidate_leng=candidate_leng,
                                                                                     instr_state=self.last_instr_state,
                                                                                     c_0=self.state_c0, step=cur_step,
                                                                                     speaker_step=speaker_step)
            else:
                with torch.no_grad():
                    vln_logit, e_reward, instra_state, progress_logits, angle_logits, region_logits, \
                    next_region_logits, target_region_logits, self.state_c0 = self.model(train_rl,
                                                                                         input_ids=mini_instr,
                                                                                         position_ids=txt_position_ids,
                                                                                         img_feat=mini_region_feat,
                                                                                         img_pos_feat=mini_region_loc,
                                                                                         orien_feat=mini_orien_feat,
                                                                                         attn_masks=attn_masks,
                                                                                         gather_index=gather_index,
                                                                                         candidate_leng=candidate_leng,
                                                                                         instr_state=self.last_instr_state,
                                                                                         c_0=self.state_c0,
                                                                                         step=cur_step,
                                                                                         speaker_step=speaker_step)

        return vln_logit, e_reward, instra_state,  candidate_leng, candidate_mask, box_num,\
               region_feat, region_loc, progress_logits, angle_logits, region_logits, \
               next_region_logits, target_region_logits

    def progress_teacher(self, step, progress_logits, ended):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        progress = np.zeros((self.env.batch_size), dtype=np.float32)
        for i, b in enumerate(self.env.batch):
            if not ended[i]:
                if self.args.task == 'NDH':
                    progress[i] = (step + 1) / len(b['shortest_path'])
                else:
                    progress[i] = (step + 1) / len(b['path'])
            else:
                progress[i] = progress_logits[i].cpu().detach().clone()

        return torch.from_numpy(progress).cuda()


    def angle_teacher(self, obs, candidate_leng, ended):
        """rel_heading: forward, backword, left, right"""
        # predict angles: forward, backward, left, right
        target = self.args.ignoreid * np.ones((len(obs), max(candidate_leng)), dtype=np.int64)
        for i, ob in enumerate(obs):
            if not ended[i]:
                for j, c in enumerate(ob['candidate']):
                    h_degree = c['heading'] * (180 / math.pi)
                    if h_degree < 0:
                        h_degree += 360
                    assert 0 <= h_degree <= 360
                    if 45 <= h_degree <= 135:
                        # right
                        target_orien = 'right'
                    elif 135 <= h_degree <= 225:
                        # backward
                        target_orien = 'backward'
                    elif 225 <= h_degree <= 315:
                        # left
                        target_orien = 'left'
                    else:
                        # forward
                        target_orien = 'forward'

                    target[i, j] = self.orien_dict[target_orien]

        target = torch.from_numpy(target).cuda()
        return target

    def candi_region_teacher(self, obs, candidate_leng, region_logits, ended):
        region_gt = np.zeros((len(obs), max(candidate_leng), 31), dtype=np.float32)
        for i, ob in enumerate(obs):
            if not ended[i]:
                for j, c in enumerate(ob['candidate']):
                    region_gt[i, j, :] = c['region_cls_gt']
                region_gt[i, j+1, :] = self.last_end_region_gt[i, :]
            else:
                region_gt[i, ...] = region_logits[i, ...].clone().detach().tolist()

        return torch.from_numpy(region_gt).cuda()

    def next_target_teacher(self, obs, ended):
        next_vp_regions = self.args.ignoreid * torch.ones(len(obs), dtype=torch.int64)
        target_vp_regions = self.args.ignoreid * torch.ones(len(obs), dtype=torch.int64)
        for i, ob in enumerate(obs):
            if not ended[i]:
                next_vp_regions[i] = ob['next_region_gt']
                target_vp_regions[i] = ob['target_region_gt']
        return next_vp_regions.cuda(), target_vp_regions.cuda()


    def rollout(self, train_ml=None, train_rl=True,  speaker=None, batch_ids=None):
        """
        :param train_ml:    The weight to train with maximum likelihood
        :param train_rl:    whether use RL in training
        :param reset:       Reset the environment
        :param speaker:     Speaker used in back translation.
                            If the speaker is not None, use back translation.
                            O.w., normal training
        :return:
        """
        if self.feedback == 'teacher' or self.feedback == 'argmax':
            train_rl = False

        # obs = np.array(self.env._get_obs())

        batch_size = self.env.batch_size

        if self.args.debug and self.args.self_train:
            print('rollout')
            if speaker:
                print(self.env.name, self.feedback, 'has speaker')
            else:
                print(self.env.name, self.feedback, 'no speaker')
        if speaker is not None:         # Trigger the self_train mode!
            noise = self.drop_env(torch.ones(self.feature_size).cuda())
            batch = self.env.batch.copy()
            speaker.env = self.env
            insts = speaker.infer_batch(featdropmask=noise)     # Use the same drop mask in speaker

            for i, (datum, inst) in enumerate(zip(batch, insts)):
                if inst[-1] != self.tok.word_to_index['<PAD>']: # The inst is not ended!
                    inst[-1] = self.tok.word_to_index['<EOS>']
                datum.pop('instructions')
                datum.pop('bert_instr_encoding')
                new_str = self.tok.decode_sentence(inst)
                datum['instructions'] = new_str
                datum['bert_instr_encoding'], datum['bert_txt_len'] = self.env.bert_tokenize(new_str.lower())
            self.env.batch = batch
            obs = np.array(self.env._get_obs())
            # obs = np.array(self.env.reset(batch))
        else:
            obs = np.array(self.env._get_obs())

        # Init the reward shaping
        last_dist = np.zeros(batch_size, np.float32)
        for i, ob in enumerate(obs):  # The init distance from the view point to the target
            last_dist[i] = ob['distance']
        if self.args.use_lstm:
            self.last_instr_state = torch.zeros((batch_size, 768)).cuda() #[None] * batch_size
        else:
            self.last_instr_state = [None] * batch_size
        # Record starting point
        traj = [{
            'angle_pred': [],
            'cur_region_pred': [],
            'next_region_pred': [],
            'target_region_pred': [],
            'instr_id': ob['instr_id'],
            'path': [(ob['viewpoint'], ob['heading'], ob['elevation'])]
        } for ob in obs]

        # For test result submission
        visited = [set() for _ in range(batch_size)]

        # Initialization the tracking state
        ended = np.array([False] * batch_size)  # Indices match permuation of the model, not env

        # Init the logs
        rewards, hidden_states, policy_log_probs, masks, entropys = [], [], [], [], []
        ml_loss, total_loss, scaler_loss = 0.,0.,0.
        self.looking_to_next_vp_feat = [] # used as representation for end viewpoint
        self.prepare_end_feat(-1, obs)
        all_e_reward,actionNum = [],0
        # acc_grad_num = math.ceil(self.episode_len / self.args.gradient_accumulation_steps)
        # instr_state_list = []
        self.last_end_region_gt = np.zeros((len(obs), 31), dtype=np.float32)
        self.state_c0 = torch.zeros((batch_size, 768)).cuda()

        for t in range(self.episode_len):
            if self.args.print_step:
                print("-- global step=%d %s t=%d local_rank=%d--"%(self.global_step, self.feedback, t, self.args.local_rank))

            # process in mini batch
            logit, e_reward, instr_state, candidate_leng, \
                   candidate_mask, box_num, region_feat, region_loc, progress_logits, \
            angle_logits, region_logits, next_region_logits, target_region_logits = \
                                                        self.process_mini_batchs(obs, train_rl, t)

            # instr_state_list.append(instr_state)
            all_e_reward.append(e_reward)

            # candidate_mask = length2mask(candidate_leng, device=self.device)
            if self.args.submit:     # Avoding cyclic path, this is actually not used
                for ob_id, ob in enumerate(obs):
                    visited[ob_id].add(ob['viewpoint'])
                    for c_id, c in enumerate(ob['candidate']):
                        if c['viewpointId'] in visited[ob_id]:
                            candidate_mask[ob_id][c_id] = 1
            logit.masked_fill_(candidate_mask, -float('inf'))

            # navigation
            target = self._teacher_action(obs, ended)

            if not train_rl:
                if self.args.angle_loss:
                    angle_target = self.angle_teacher(obs, candidate_leng, ended)

                if self.args.progress_loss:
                    progress_target = self.progress_teacher(t, progress_logits, ended)

                if self.args.candi_region_loss:
                    region_target = self.candi_region_teacher(obs, candidate_leng, region_logits, ended)

                if self.args.next_region_loss or self.args.target_region_loss:
                    next_vp_regions_gt, target_vp_regions_gt = self.next_target_teacher(obs, ended)

            # Supervised training
            if self.feedback == 'teacher':
                nav_loss = self.action_crit(logit, target)
                self.logs['nav_loss'] += nav_loss.item()
                ml_loss += nav_loss

                if self.args.progress_loss:
                    progress_loss = self.progress_crit(progress_logits, progress_target)
                    self.logs['progress_loss'] += progress_loss.item()
                    ml_loss += progress_loss

                if self.args.angle_loss:
                    angle_loss = self.angle_crit(angle_logits.reshape(-1, angle_logits.size()[-1]), angle_target.reshape(-1))
                    self.logs['angle_loss'] += angle_loss.item()
                    ml_loss += angle_loss

                if self.args.candi_region_loss:
                    region_loss = self.region_crit(region_logits, region_target)
                    self.logs['region_loss'] += region_loss.item()
                    ml_loss += region_loss

                if self.args.next_region_loss:
                    next_region_loss = self.single_region_crit(next_region_logits, next_vp_regions_gt)
                    self.logs['next_region_loss'] += next_region_loss.item()
                    ml_loss += self.args.next_region_pred_w * next_region_loss

                if self.args.target_region_loss:
                    target_region_loss = self.single_region_crit(target_region_logits, target_vp_regions_gt)
                    self.logs['target_region_loss'] += target_region_loss.item()
                    ml_loss += self.args.target_region_pred_w * target_region_loss


            if self.feedback == 'teacher':
                a_t = target  # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = logit.max(1)  # student forcing - argmax
                a_t = a_t.detach()
                if self.phase=='train':
                    log_probs = F.log_softmax(logit, 1)  # Calculate the log_prob here
                    policy_log_probs.append(log_probs.gather(1, a_t.unsqueeze(1)))  # Gather the log_prob for each batch

                if self.args.angle_loss:
                    _, angle_pred = angle_logits.max(2)
                    angle_pred = angle_pred.detach()

                if self.args.next_region_loss:
                    _, next_region_pred = next_region_logits.max(1)
                    next_region_pred = next_region_pred.detach()

                if self.args.target_region_loss:
                    _, target_region_pred = target_region_logits.max(1)
                    target_region_pred = target_region_pred.detach()

                for i in range(self.env.batch_size):
                    if not ended[i]:
                        cand_len = candidate_leng[i]
                        if self.args.candi_region_loss:
                            cur_region_pred = 1 * (region_logits > 0.5)
                            temp_region_target = 1 * (region_target > 0.5)
                            cur_region_inter = cur_region_pred & temp_region_target
                            cur_region_union = cur_region_pred | temp_region_target
                            batch_iou = torch.sum(cur_region_inter[i, :cand_len, :], dim=1).float() / (
                                        torch.sum(cur_region_union[i, :cand_len, :], dim=1) + 1e-12)
                            traj[i]['cur_region_pred'] += batch_iou.tolist()

                        if self.args.angle_loss:
                            angle_res = 1 * (angle_pred[i][:cand_len-1] == angle_target[i][:cand_len-1])
                            traj[i]['angle_pred'] += angle_res.tolist()

                        if self.args.next_region_loss:
                            next_region_res = 1 * (next_region_pred[i] == next_vp_regions_gt[i])
                            traj[i]['next_region_pred'].append(next_region_res.item())

                        if self.args.target_region_loss:
                            target_region_res = 1 * (target_region_pred[i] == target_vp_regions_gt[i])
                            traj[i]['target_region_pred'].append(target_region_res.item())



            elif self.feedback == 'sample':
                probs = F.softmax(logit, 1)  # sampling an action from model
                c = torch.distributions.Categorical(probs)
                self.logs['entropy'] += c.entropy().sum().item()  # For log
                entropys.append(c.entropy())  # For optimization
                a_t = c.sample().detach()
                # check a_t, avoid stop at 1st step
                if t==0:
                    for i_, a_t_0 in enumerate(a_t):
                        if a_t_0 == candidate_leng[i_] - 1:
                            a_t[i_] = 0

                policy_log_probs.append(c.log_prob(a_t))
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Prepare environment action
            # NOTE: Env action is in the perm_obs space
            cpu_a_t = a_t.cpu().numpy()
            for i, next_id in enumerate(cpu_a_t):
                if next_id == (candidate_leng[i] - 1) or next_id == self.args.ignoreid:  # The last action is <end>
                    cpu_a_t[i] = -1  # Change the <end> and ignore action to -1

            # update representation for ending
            self.prepare_end_feat(t, obs, cpu_a_t, candidate_leng, box_num, region_feat, region_loc)


            # temp_instr_state = []
            if self.args.no_history_state==False:
                if self.args.use_lstm:
                    for i, act in enumerate(cpu_a_t):
                        if act != -1:
                            self.last_instr_state[i, :] = instr_state[i,:]
                            self.last_end_region_gt[i, :] = obs[i]['candidate'][cpu_a_t[i]]['region_cls_gt']
                else:
                    for i, act in enumerate(cpu_a_t):
                        if act != -1:
                            self.last_instr_state[i] = instr_state[sum(candidate_leng[:i])+act,:]
                            self.last_end_region_gt[i, :] = obs[i]['candidate'][cpu_a_t[i]]['region_cls_gt']

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, obs, None, traj)
            if self.args.verbose:
                start = time.time()
                obs = np.array(self.env._get_obs())
                p2 = time.time()
                if self.args.local_rank==0:
                    print('get obs time: %f' % (p2 - start))
            else:
                obs = np.array(self.env._get_obs())

            # Calculate the mask and reward
            masks.append(np.array(1.0-ended))

            if train_rl:
                self.logs['rl_step'] += 1 # reset every log_every
                dist = np.zeros(batch_size, np.float32)
                reward = np.zeros(batch_size, np.float32)
                for i, ob in enumerate(obs):
                    dist[i] = ob['distance']
                    if ended[i]:  # If the action is already finished BEFORE THIS ACTION.
                        reward[i] = 0.
                    else:  # Calculate the reward
                        actionNum += 1
                        action_idx = cpu_a_t[i]
                        if action_idx == -1:  # If the action now is end
                            if dist[i] < 3:  # Correct
                                reward[i] = 2.
                            else:  # Incorrect
                                reward[i] = -2.
                        else:  # The action is not end
                            reward[i] = - (dist[i] - last_dist[i])  # Change of distance
                            if reward[i] > 0:  # Quantification
                                reward[i] = 1
                            elif reward[i] < 0:
                                reward[i] = -1
                            else:
                                raise NameError("The action doesn't change the move")
                rewards.append(reward)
                last_dist[:] = dist
            else:
                if self.phase=='train':
                    self.logs['ml_step'] += 1 # reset every log_every
                for i in range(batch_size):
                    if ended[i] == False:
                        actionNum += 1
            # Update the finished actions
            # -1 means ended or ignored (already ended)
            ended[:] = np.logical_or(ended, (cpu_a_t == -1))

            # Early exit if all ended
            if ended.all():
                break

        if train_rl:
            # Last action in A2C
            _, last_e_reward, _, _, _,  _, _, _, _, _, _, _, _ = \
                self.process_mini_batchs(obs, train_rl, self.episode_len)

            rl_loss = 0.
            # NOW, A2C!!!
            # Calculate the final discounted reward
            last_value__ =  last_e_reward.detach()    # The value esti of the last state, remove the grad for safety
            discount_reward = np.zeros(batch_size, np.float32)  # The inital reward is zero
            for i in range(batch_size):
                if not ended[i]:        # If the action is not ended, use the value function as the last reward
                    discount_reward[i] = last_value__[i]

            length = len(rewards)

            for t in range(length-1, -1, -1):
                discount_reward = discount_reward * self.args.gamma + rewards[t]   # If it ended, the reward will be 0
                mask_ = Variable(torch.from_numpy(masks[t]), requires_grad=False).cuda()
                clip_reward = discount_reward.copy()
                r_ = Variable(torch.from_numpy(clip_reward), requires_grad=False).cuda()
                v_ = all_e_reward[t]
                a_ = (r_ - v_).detach()

                # r_: The higher, the better. -ln(p(action)) * (discount_reward - value)
                rl_loss += (-policy_log_probs[t] * a_ * mask_).sum()
                tmp_rl_loss = (((r_ - v_) ** 2) * mask_).sum()
                self.logs['critic_loss'] += tmp_rl_loss.item()
                rl_loss += tmp_rl_loss * 0.5  # 1/2 L2 loss
                if self.feedback == 'sample':
                    rl_loss += (- 0.01 * entropys[t] * mask_).sum()

            self.logs['rl_loss'] += rl_loss.item()

            if self.args.normalize_loss == 'total':
                rl_loss /= actionNum
            elif self.args.normalize_loss == 'batch':
                rl_loss /= batch_size
            else:
                assert self.args.normalize_loss == 'none'

        if self.phase == 'train':
            if train_ml is not None:
                self.logs['ml_loss'] += ml_loss.item()
                self.loss += ml_loss * train_ml / batch_size
                assert self.args.gradient_accumulation_steps <= 0
            else:
                self.loss += rl_loss / batch_size

        if self.phase == 'train':
            if train_ml is not None:
                self.logs['ml_action_num'] += actionNum
            else:
                self.logs['rl_action_num'] += actionNum
        else:
            self.logs['eval_action_num'] += actionNum

        return traj


    def prepare_end_feat(self, t, perm_obs, cpu_a_t=None, candidate_leng=None,
                                box_num=None, region_feat=None, region_loc=None):
        candi_index = 0
        if t==-1:# check region_loc is real size as region_num, but angel_feat and region_feat not
            for i, ob in enumerate(perm_obs): # initialization, should be one less likely be selected
                self.looking_to_next_vp_feat.append({ 'region_num':1,
                               'angle_feat':np.zeros((1,4),dtype=np.float32), #check here
                               'region_feat':np.zeros((self.env.bert_max_region_num, 2048),dtype=np.float32),
                               'region_loc':np.array([0, 0, 1, 1, 1, 1, 1],dtype=np.float32)
                })
                self.looking_to_next_vp_feat[i]['region_feat'][0, :] = np.random.rand(1, 2048).astype(np.float32)
        else:
            for i, ob in enumerate(perm_obs):
                if cpu_a_t[i] == -1:
                    pass # keep unchanged

                else:
                    candi_index += cpu_a_t[i]
                    self.looking_to_next_vp_feat[i]['region_num'] = box_num[candi_index]
                    heading_angle = math.pi if self.args.use_angle_pi else 0
                    self.looking_to_next_vp_feat[i]['angle_feat'] = angle_feature(heading_angle,0,self.args.angle_feat_size)
                    self.looking_to_next_vp_feat[i]['region_feat'] = region_feat[candi_index, :, :]
                    self.looking_to_next_vp_feat[i]['region_loc'] = region_loc[candi_index, :, :]

                candi_index = np.sum(candidate_leng[0:i+1])

    def test(self, use_dropout=False, feedback='argmax', allow_cheat=False, iters=None):
        ''' Evaluate once on each instruction in the current environment '''
        self.feedback = feedback
        super(BertAgent, self).test(iters)

    def train_with_grad_accumulate(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback
        batch_ids = kwargs['batch_ids']
        assert feedback == 'sample'

        if self.args.ml_weight != 0:
            self.loss = 0.0
            self.feedback = 'teacher'
            self.env.prepare_batch(batch_ids[:int(0.5 * len(batch_ids))])

            self.rollout(train_ml=self.args.ml_weight, train_rl=False, **kwargs)
            self.loss.backward()

            self.loss = 0.0
            self.feedback = 'sample'
            self.env.prepare_batch(batch_ids[int(0.5*len(batch_ids)):])
            self.rollout(train_ml=None, train_rl=True, **kwargs)

        else:
            self.feedback = 'sample'
            self.env.prepare_batch(batch_ids)
            self.rollout(train_ml=None, train_rl=True, **kwargs)


    def train(self, n_iters, feedback='teacher', **kwargs):
        ''' Train for a given number of iterations '''
        self.feedback = feedback
        batch_ids = kwargs['batch_ids']
        if feedback == 'teacher':
            assert False
            self.feedback = 'teacher'
            self.env.prepare_batch(batch_ids)
            self.rollout(train_ml=self.args.teacher_weight, train_rl=False,  **kwargs)
        elif feedback == 'sample':
            if self.args.ml_weight != 0:
                self.feedback = 'teacher'
                self.env.prepare_batch(batch_ids[:int(0.5*len(batch_ids))])
                self.rollout(train_ml=self.args.ml_weight, train_rl=False,  **kwargs)
                self.feedback = 'sample'
                self.env.prepare_batch(batch_ids[int(0.5*len(batch_ids)):])
                self.rollout(train_ml=None, train_rl=True, **kwargs)
            else:
                self.feedback = 'sample'
                self.env.prepare_batch(batch_ids)
                self.rollout(train_ml=None, train_rl=True, **kwargs)
        else:
            assert False

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        if not os.path.exists(the_dir):
            os.makedirs(the_dir)

        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in self.model.state_dict().items()}
        torch.save(
            {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "epoch_id": epoch,
                "best_val": self.best_val,
                "global_step": self.global_step,
                "logs": self.logs
            },
            path)

