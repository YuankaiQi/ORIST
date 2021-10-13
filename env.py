''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
from tqdm import tqdm
import numpy as np
import math
import copy
import torch
import utils
import os
import random
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from r2r_utils import (load_datasets, load_nav_graphs,angle_feature,read_img_features_h5,read_h5_vp_feat,
                       read_h5_cache,read_aug_path_cache,get_all_point_angle_feature,new_simulator)

import json
from utils.logger import LOGGER


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
            self.feature_size = 2048
        if feature_store:
            self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for j in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setNavGraphPath('connectivity')
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)


    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0) #change here for ddp
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                feature_states.append((feature, state))
            else:
                feature_states.append((None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class BaseBatch():
    ''' Base class for navigation tasks, using discretized viewpoints and pretrained features '''
    buffered_state_dict, buffered_candi_vp_2_ids = [], []
    vp_feature_cache = []
    region_feature_cache = {}
    cache_initialized = False

    def __init__(self, feature_store, batch_size=100, seed=0, splits=['train'], tokenizer=None,
                 name=None, bert_tokenizer=None, bert_padding_index=None, args=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.feature_size = self.env.feature_size
        self.data = []
        self.args = args
        if tokenizer:
            self.tok = tokenizer
        if bert_tokenizer:
            self.bert_tok = bert_tokenizer
        self.bert_padding_index = bert_padding_index
        self.bert_max_region_num = args.max_bb
        self.bert_max_seq_length = args.max_txt_len

        scans = []
        for split in splits:
            for item in load_datasets([split], args.task):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']): #check blow lines
                    new_item = dict(item) # maybe need to decompose a task with multiple paths to severl items
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j) # how to handle multiple path
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if bert_tokenizer:
                        new_item['bert_instr_encoding'], new_item['bert_txt_len'] \
                             = self.bert_tokenize(instr.lower()) #make sure instr in lower case

                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.index2instr_id = {}
        for i_, item_ in enumerate(self.data):
            self.index2instr_id[i_] = item_['instr_id']

        self.ix = 0
        self.batch_size = batch_size
        self.batch = [0]*batch_size
        self._load_nav_graphs()

        self.angle_feature = get_all_point_angle_feature(args.angle_feat_size)
        self.sim = new_simulator()

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        self.data_len = len(self.data)
        print('BaseBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def bert_tokenize(self, instr):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        
        tokens = []
        tokens.extend(self.bert_tok.convert_tokens_to_ids(['[CLS]']))
        for word in instr.strip().split():
            # if word in self.bert_tok.vocab:
            #     tokens.extend([self.bert_tok.vocab[word]])
            # else:
            #     tokens.extend([self.bert_tok.vocab['[UNK]']])
            if word == '[sep]':
                tokens.extend([self.bert_tok.vocab['[SEP]']])
            else:
                ws = self.bert_tok.tokenize(word)
                if not ws:
                    # some special char
                    continue
                tokens.extend(self.bert_tok.convert_tokens_to_ids(ws))
        if len(tokens)>self.bert_max_seq_length-1:
            tokens = tokens[:self.bert_max_seq_length-1]
        tokens.extend(self.bert_tok.convert_tokens_to_ids(['[SEP]']))

        txt_len = len(tokens)
        if txt_len < self.bert_max_seq_length:
            # Note here we pad in front of the sentence
            padding = [self.bert_padding_index] * (self.bert_max_seq_length - txt_len)
            tokens = tokens + padding


        assert len(tokens)==self.bert_max_seq_length

        return tokens, txt_len

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _set_batch_id(self,id):
        self.ix = id

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId, ix):
        base_heading = (viewId % 12) * math.radians(30)
        long_id = "%s_%s" % (scanId, viewpointId)
        candidate = BaseBatch.buffered_state_dict[long_id]
        candidate_new = []
        for c in candidate:
            c_new = c.copy()
            ix = c_new['pointId']
            normalized_heading = c_new['normalized_heading']
            visual_feat = feature[ix] if feature else np.zeros(2176)
            c_new['heading'] = normalized_heading - base_heading
            angle_feat = angle_feature(c_new['heading'], c_new['elevation'],self.args.angle_feat_size)
            c_new['angle_feat'] = angle_feat
            c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)

            rf_id = '%s_%02d' % (long_id, ix)
            # default 25 objests, with descend dection score
            c_new['region_feat'], c_new['region_loc'], c_new['region_num'],region_labels \
                = BaseBatch.region_feature_cache[rf_id]
            c_new['region_feat'] = c_new['region_feat'][:self.bert_max_region_num,:]
            c_new['region_loc'] = c_new['region_loc'][:self.bert_max_region_num, :]

            if c_new['region_num'] > self.bert_max_region_num:
                c_new['region_num'] = self.bert_max_region_num
            candidate_new.append(c_new)

        return candidate_new

    def get_imgs(self):
        imgs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            img = np.array(state.rgb, copy=True)
            img = img[:,:,::-1]
            imgs.append(img)
        return imgs

    def prepare_batch(self,batch_ids):
        '''
        Set a new batch, this function might be rewrite for HANNA and other navigation tasks
        '''
        for i, bid in enumerate(batch_ids):
            if bid>=self.data_len:
                bid = self.data_len-1
            self.batch[i] = self.data[bid]
        scanIds = [item['scan'] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)

    def _get_obs(self):
        '''
        Get observations at the current step, this function might be rewrite for HANNA and other navigation tasks
        '''
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i] #change here, index i should be passwd in from each process
            base_view_id = state.viewIndex # current heading
            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex, i)
            candi_vp_2_id = BaseBatch.buffered_candi_vp_2_ids['%s_%s'%(state.scanId, state.location.viewpointId)]
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1) if feature else self.angle_feature[base_view_id]

            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'teacher': self._shortest_path_action(state, item['path'][-1]),
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'path_id' : item['path_id'],
                'candi_vp_2_id': candi_vp_2_id
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            if 'bert_instr_encoding' in item:
                obs[-1]['bert_instr_encoding'] = item['bert_instr_encoding']
                obs[-1]['bert_txt_len'] = item['bert_txt_len']
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''
    buffered_state_dict, buffered_candi_vp_2_ids = [],[]
    buffered_r2r = [], {}
    cache_initialized = False

    all_graphs, all_paths, all_distances = {}, {}, {}
    aug_path_cache = {}
    region_cls_gt = {}
    house_pano_info = {}

    def __init__(self, feature_store, batch_size=100, seed=0, splits=['train'], tokenizer=None,
                 name=None, bert_tokenizer=None, bert_padding_index=None, args=None):
        self.env = EnvBatch(feature_store=feature_store, batch_size=batch_size)
        self.feature_size = self.env.feature_size
        self.data = []
        self.args = args

        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        if R2RBatch.cache_initialized==False:
            self._load_all_graphs()
            R2RBatch.buffered_state_dict, R2RBatch.buffered_candi_vp_2_ids = \
                                                read_h5_cache(args.candi_whole_feat_cache_dir, 2)
            with open(args.region_cls_gt, 'r') as f:
                LOGGER.info('Load region cls gt from %s' % args.region_cls_gt)
                R2RBatch.region_cls_gt = json.load(f)

            with open(args.house_pano_info, 'r') as f:
                LOGGER.info('Load house pano info from %s' % args.house_pano_info)
                R2RBatch.house_pano_info = json.load(f)
            if not args.debug:
                R2RBatch.region_feature_cache = read_h5_cache(args.candi_region_feat_cache_dir, 1)

            R2RBatch.cache_initialized = True
        self.cache_candidate_region_id = {} 
        if tokenizer:
            self.tok = tokenizer
        if bert_tokenizer:
            self.bert_tok = bert_tokenizer
        self.bert_padding_index = bert_padding_index
        self.bert_max_region_num = args.max_bb
        self.bert_max_seq_length = args.max_txt_len
        scans = []
        step = 0
        for split in splits:
            for item in tqdm(load_datasets([split], args.task)):
                if args.debug and step > 500:
                    break
                step += 1
                # Split multiple instructions into separate entries
                for j, instr in enumerate(item['instructions']):
                    new_item = dict(item)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)

                    # for ndh
                    if args.task == 'NDH':
                        new_item['instr_id'] = str(item['path_id'])
                        # Default path type is 'mixed', which is equal to 'trusted_path' in the original paper
                        if args.path_type == 'oracle':
                            new_item['path'] = item['planner_path']
                        elif args.path_type == 'navigator':
                            new_item['path'] = item['player_path']
                        else:
                            # mixed path
                            pass

                        if bert_tokenizer:
                            new_item['bert_instr_encoding'], new_item['bert_txt_len'] \
                                = self.bert_tokenize_ndh(instr.lower())  # make sure instr in lower case
                    elif args.task == 'R2R':
                        new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                        if bert_tokenizer:
                            new_item['bert_instr_encoding'], new_item['bert_txt_len'] \
                                = self.bert_tokenize(instr.lower())  # make sure instr in lower case
                    elif args.task == 'REVERIE':
                        new_item['instr_id'] = '%s_%d' % (item['id'], j)
                        if 'path_id' not in new_item:
                            new_item['path_id'] = None
                        if bert_tokenizer:
                            new_item['bert_instr_encoding'], new_item['bert_txt_len'] \
                                = self.bert_tokenize(instr.lower())  # make sure instr in lower case
                    else:
                        assert 1==0

                    # shortest path for progress
                    item_path = new_item['path']
                    new_item['shortest_path'] = R2RBatch.all_paths[new_item['scan']][item_path[0]][item_path[-1]]

                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        if splits[0]=='train': # avoid step at the second step
                            if len(new_item['shortest_path'])>2: # avoid
                                self.data.append(new_item)
                                scans.append(item['scan'])
                        else:
                            self.data.append(new_item)
                            scans.append(item['scan'])

        self.scans = set(scans)
        self.splits = splits
        self.index2instr_id = {}
        for i_, item_ in enumerate(self.data):
            self.index2instr_id[i_] = item_['instr_id']

        self.ix = 0
        self.batch_size = batch_size
        self.batch = [0] * batch_size
        self._load_nav_graphs()

        self.angle_feature = get_all_point_angle_feature(args.angle_feat_size)
        self.speaker_angle_feature = get_all_point_angle_feature(args.speaker_angle_feat_size)
        self.sim = new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        self.data_len = len(self.data)
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def bert_tokenize_ndh(self, instr):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        tokens = []
        tokens.extend(self.bert_tok.convert_tokens_to_ids(['[CLS]']))
        for word in instr.strip().split():
            if word == '[sep]':
                tokens.extend(self.bert_tok.convert_tokens_to_ids(['[SEP]']))
            else:
                # if word in self.bert_tok.vocab:
                #     tokens.extend([self.bert_tok.vocab[word]])
                # else:
                #     tokens.extend([self.bert_tok.vocab['[UNK]']])
                ws = self.bert_tok.tokenize(word)
                if not ws:
                    # some special char
                    continue
                tokens.extend(self.bert_tok.convert_tokens_to_ids(ws))
        if len(tokens) > self.bert_max_seq_length-1:
            tokens = tokens[:self.bert_max_seq_length-1]
        tokens.extend(self.bert_tok.convert_tokens_to_ids(['[SEP]']))

        txt_len = len(tokens)
        if txt_len < self.bert_max_seq_length:
            # Note here we pad in front of the sentence
            padding = [self.bert_padding_index] * (self.bert_max_seq_length - txt_len)
            tokens = tokens + padding

        assert len(tokens)==self.bert_max_seq_length

        return tokens, txt_len

    def bert_tokenize(self, instr):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        tokens = []
        tokens.extend(self.bert_tok.convert_tokens_to_ids(['[CLS]']))
        for word in instr.strip().split():
            # if word in self.bert_tok.vocab:
            #     tokens.extend([self.bert_tok.vocab[word]])
            # else:
            #     tokens.extend([self.bert_tok.vocab['[UNK]']])
            if word == '[sep]':
                tokens.extend([self.bert_tok.vocab['[SEP]']])
            else:
                ws = self.bert_tok.tokenize(word)
                if not ws:
                    # some special char
                    continue
                tokens.extend(self.bert_tok.convert_tokens_to_ids(ws))
            # below is original version
            # ws = self.bert_tok.tokenize(word)
            # if not ws:
            #     # some special char
            #     continue
            # tokens.extend(self.bert_tok.convert_tokens_to_ids(ws))
        if len(tokens) > self.bert_max_seq_length-1:
            tokens = tokens[:self.bert_max_seq_length-1]
        tokens.extend(self.bert_tok.convert_tokens_to_ids(['[SEP]']))

        txt_len = len(tokens)
        if txt_len < self.bert_max_seq_length:
            # Note here we pad in front of the sentence
            padding = [self.bert_padding_index] * (self.bert_max_seq_length - txt_len)
            tokens = tokens + padding

        assert len(tokens)==self.bert_max_seq_length

        return tokens, txt_len

    def size(self):
        return len(self.data)

    def _load_all_graphs(self):
        # print('Loading all navigation graphs' % len(self.scans))
        with open('connectivity/scans.txt', 'r') as f:
            scans = [scan.strip() for scan in f.readlines()]

        all_graphs = load_nav_graphs(scans)
        all_paths = {}
        for scan, G in all_graphs.items(): # compute all shortest paths
            all_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))

        all_distances = {}
        for scan, G in all_graphs.items(): # compute all shortest paths
            all_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        R2RBatch.all_graphs = all_graphs
        R2RBatch.all_paths = all_paths
        R2RBatch.all_distances = all_distances

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """

        if R2RBatch.cache_initialized==False:
            self._load_all_graphs()

        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = {}
        self.paths = {}
        self.distances = {}
        for scan in self.scans:
            self.graphs[scan] = R2RBatch.all_graphs[scan]
            self.paths[scan] = R2RBatch.all_paths[scan]
            self.distances[scan] = R2RBatch.all_distances[scan]

        # self.graphs = load_nav_graphs(self.scans)
        # self.paths = {}
        # for scan, G in self.graphs.items(): # compute all shortest paths
        #     self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        # self.distances = {}
        # for scan, G in self.graphs.items(): # compute all shortest paths
        #     self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    # def _load_nav_graphs(self):
    #     """
    #     load graph from self.scan,
    #     Store the graph {scan_id: graph} in self.graphs
    #     Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
    #     Store the distances in self.distances. (Structure see above)
    #     Load connectivity graph for each scan, useful for reasoning about shortest paths
    #     :return: None
    #     """
    #     print('Loading navigation graphs for %d scans' % len(self.scans))
    #     self.graphs = load_nav_graphs(self.scans)
    #     self.paths = {}
    #     for scan, G in self.graphs.items(): # compute all shortest paths
    #         self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    #     self.distances = {}
    #     for scan, G in self.graphs.items(): # compute all shortest paths
    #         self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _set_batch_id(self,id):
        self.ix = id

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, scanId, viewpointId, viewId, ix, use_speaker=None):
        base_heading = (viewId % 12) * math.radians(30)
        long_id = "%s_%s" % (scanId, viewpointId)
        candidate = R2RBatch.buffered_state_dict[long_id]
        candidate_new = []
        # item = self.batch[ix]
        for c in candidate:
            c_new = c.copy()
            ix = c_new['pointId']
            normalized_heading = c_new['normalized_heading']
            if use_speaker==None:
                if feature is not None:
                    visual_feat = feature[ix]
                else:
                    visual_feat = np.zeros(2048) #2176
            elif use_speaker=='wholeImg':
                visual_feat = feature[ix]
            elif use_speaker=='region':
                rf_id = '%s_%02d' % (long_id, ix)
                region_feat, _, region_num, region_labels \
                    = R2RBatch.region_feature_cache[rf_id]
                objId = np.random.randint(region_num)
                visual_feat = region_feat[objId]
            else:
                assert 1==0

            c_new['heading'] = normalized_heading - base_heading
            angle_feat = angle_feature(c_new['heading'], c_new['elevation'],self.args.angle_feat_size)
            c_new['angle_feat'] = angle_feat
            c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)

            rf_id = '%s_%02d' % (long_id, ix)

            c_new['img_id'] = rf_id
            c_new['region_cls_gt'] = copy.deepcopy(R2RBatch.region_cls_gt[rf_id])
            c_new['region_cls_gt'][R2RBatch.house_pano_info[scanId][c_new['viewpointId']]] = 1
            # default 30 objests, with descend dection score

            if self.args.debug:
                c_new['region_feat'] = np.zeros((5, 2048))
                c_new['region_loc'] = np.zeros((5, 7))
                c_new['region_num'] = 5
            else:
                c_new['region_feat'], c_new['region_loc'], c_new['region_num'], region_labels \
                    = R2RBatch.region_feature_cache[rf_id]

                if self.args.add_whole_img_feat:

                    if c_new['region_num'] >= self.bert_max_region_num:
                        c_new['region_num'] = self.bert_max_region_num-1
                        c_new['region_feat'] = c_new['region_feat'][:self.bert_max_region_num-1, :]
                        c_new['region_loc'] = c_new['region_loc'][:self.bert_max_region_num-1, :]


                    assert c_new['region_num'] == c_new['region_feat'].shape[0] == c_new['region_loc'].shape[0]
                    c_new['region_feat'] = np.concatenate(
                        (c_new['region_feat'], visual_feat.reshape(-1, self.feature_size)), axis=0)
                    c_new['region_loc'] = np.concatenate(
                        (c_new['region_loc'], np.array([[0, 0, 1, 1, 1, 1, 1]], dtype=np.float32)), axis=0)
                    c_new['region_num'] += 1
                else:
                    c_new['region_feat'] = c_new['region_feat'][:self.bert_max_region_num, :]
                    c_new['region_loc'] = c_new['region_loc'][:self.bert_max_region_num, :]
                    if c_new['region_num'] > self.bert_max_region_num:
                        c_new['region_num'] = self.bert_max_region_num

            # instr_rf_id = '%s_%s'%(item['instr_id'],rf_id)
            # if instr_rf_id in self.cache_candidate_region_id:
            #     ids = self.cache_candidate_region_id[instr_rf_id]
            # else:
            #     ids = self.calc_topk_matched_regions(item['instructions'],region_labels)
            #     self.cache_candidate_region_id[instr_rf_id] = ids
            # c_new['region_feat'] = c_new['region_feat'][ids,:]  # check whether affect R2RBatch.region_feature_cache[rf_id]
            # c_new['region_loc'] = c_new['region_loc'][ids, :]
            # c_new['region_num'] = len(ids)
            # c_new.pop('normalized_heading')


            candidate_new.append(c_new)

        return candidate_new

    def calc_topk_matched_regions(self, instra, region_labels):
        values=[]
        tokens = self.tok.split_sentence(instra)
        for token in tokens:
            if token in R2RBatch.glove_embs:
                values.append(R2RBatch.glove_embs[token]/np.linalg.norm(R2RBatch.glove_embs[token]))
            else:
                new_tok = Word(token).singularize()
                if new_tok in R2RBatch.glove_embs:
                    values.append(R2RBatch.glove_embs[new_tok]/np.linalg.norm(R2RBatch.glove_embs[new_tok]))
                else:
                    new_tok = Word(token).correct()
                    if new_tok in R2RBatch.glove_embs:
                        values.append(R2RBatch.glove_embs[new_tok]/np.linalg.norm(R2RBatch.glove_embs[new_tok]))
        values = np.array(values)
        similarity = []
        for labels in region_labels:
            tokens = labels.decode('ascii').split('#')
            query = []
            for token in tokens:
                if token in R2RBatch.glove_embs:
                    query.append(R2RBatch.glove_embs[token]/np.linalg.norm(R2RBatch.glove_embs[token]))
                else:
                    new_tok = Word(token).singularize()
                    if new_tok in R2RBatch.glove_embs:
                        query.append(R2RBatch.glove_embs[new_tok]/np.linalg.norm(R2RBatch.glove_embs[new_tok]))
                    else:
                        new_tok = Word(token).correct()
                        if new_tok in R2RBatch.glove_embs:
                            query.append(R2RBatch.glove_embs[new_tok]/np.linalg.norm(R2RBatch.glove_embs[new_tok]))
            if len(query)==0:
                similarity.append(0)
            else:
                query = np.array(query) # check dims, should be 2d matrix
                temp_score = np.matmul(query,np.transpose(values)) #kx300, 300xn=kxn
                similarity.append(np.average(temp_score.max(axis=1))) # get max of each row, then mean
        inds = np.argsort(similarity) # ascend sort

        return inds[-self.bert_max_region_num:]
        
    def get_imgs(self):
        imgs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex  # current heading
            img = np.array(state.rgb, copy=True)
            img = img[:,:,::-1]
            imgs.append(img)

        return imgs

    def prepare_batch(self,batch_ids):
        for i, bid in enumerate(batch_ids):
            if bid>=self.data_len:
                bid = self.data_len-1
            self.batch[i] = self.data[bid]
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)

    def _get_obs(self, speaker=False):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i] #change here, index i should be passwd in from each process
            base_view_id = state.viewIndex # current heading

            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex, i)
            candi_vp_2_id = R2RBatch.buffered_candi_vp_2_ids['%s_%s'%(state.scanId, state.location.viewpointId)]
            # (visual_feature, angel_feature) for views
            if feature is not None:
                feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            else:
                feature = self.angle_feature[base_view_id]
                # print('Note feature does not include vis feature in R2RBatch._get_obs()')
            teacher_vp = self._shortest_path_action(state, item['path'][-1])
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher': teacher_vp,
                'path_id' : item['path_id'], # check whether need to change to r2r's
                'candi_vp_2_id': candi_vp_2_id
            })
            obs[-1]['next_region_gt'] = self.house_pano_info[state.scanId][teacher_vp]
            obs[-1]['target_region_gt'] = self.house_pano_info[state.scanId][item['path'][-1]]
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            if 'bert_instr_encoding' in item:
                obs[-1]['bert_instr_encoding'] = item['bert_instr_encoding']
                obs[-1]['bert_txt_len'] = item['bert_txt_len']
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]

        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats
