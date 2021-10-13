''' Evaluation of agent trajectories '''

import json
import os
import sys
import torch
from collections import defaultdict
import networkx as nx
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

from env import R2RBatch
from r2r_utils import load_datasets, load_nav_graphs



class MultiEvaluationNDH(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok, args=None):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.args = args
        self.gt_path_type = 'path'
        if self.args.path_type == 'oracle':
            self.gt_path_type = 'planner_path'
        elif args.path_type == 'navigator':
            self.gt_path_type = 'player_path'
        else:
            # mixed path
            pass
        for split in splits:
            for item in load_datasets([split], args.task):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['path_id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids.append(str(item['path_id']))
                # self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, item):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        instr_id, path = item['inst_idx'], item['trajectory']
        gt = self.gt[instr_id]

        # fix a bug
        start = gt[self.gt_path_type][0]
        if start != path[0][0]:
            print(instr_id)
            print(gt)
            print(path)
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt[self.gt_path_type][-1]

        planner_goal = gt['planner_path'][-1]
        final_position = path[-1][0]  # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        nearest_planner_position = self._get_nearest(gt['scan'], planner_goal, path)
        dist_to_end_start = None
        dist_to_end_end = None
        for end_pano in gt['end_panos']:
            d = self.distances[gt['scan']][start][end_pano]
            if dist_to_end_start is None or d < dist_to_end_start:
                dist_to_end_start = d
            d = self.distances[gt['scan']][final_position][end_pano]
            if dist_to_end_end is None or d < dist_to_end_end:
                dist_to_end_end = d

        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['oracle_plan_errors'].append(self.distances[gt['scan']][nearest_planner_position][planner_goal])
        self.scores['dist_to_end_reductions'].append(dist_to_end_start - dist_to_end_end)
        # self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0  # Work out the length of the path in meters
        prev = path[0]
        for curr in path[1:]:
            if prev[0] != curr[0]:
                try:
                    self.graphs[gt['scan']][prev[0]][curr[0]]
                except KeyError as err:
                    print('Error: The provided trajectory moves from %s to %s but the navigation graph contains no ' \
                          'edge between these viewpoints. Please ensure the provided navigation trajectories ' \
                          'are valid, so that trajectory length can be accurately calculated.' % (prev[0], curr[0]))
                    raise
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            prev = curr
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])

        self.scores['num_angle_pred'].append(len(item['angle_pred']))
        self.scores['num_region_pred'].append(len(item['target_region_pred']))
        self.scores['num_candi_region_pred'].append(len(item['cur_region_pred']))


        self.scores['angle_acc'].append(sum(item['angle_pred']))
        self.scores['next_region_acc'].append(sum(item['next_region_pred']))
        self.scores['target_region_acc'].append(sum(item['target_region_pred']))
        self.scores['cur_region_iou'].append(sum(item['cur_region_pred']))


    def distributed_score(self, output_file, repeated_task_id):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        self.repeated_score = None
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids

            # instr_idx to inst_idx,
            if item['inst_idx'] in instr_ids:
                # instr_ids.remove(item['inst_idx'])
                self._score_item(item)
                if item['inst_idx'] == repeated_task_id:
                    self.repeated_score = {}
                    for key in self.scores.keys():
                        self.repeated_score[key] = self.scores[key][-1]

        # multi
        num_angle_pred = np.sum(self.scores['num_angle_pred'])
        num_region_pred = np.sum(self.scores['num_region_pred'])
        num_candi_region_pred = np.sum(self.scores['num_candi_region_pred'])

        angle_acc = np.sum(self.scores['angle_acc'])
        next_region_acc = np.sum(self.scores['next_region_acc'])
        target_region_acc = np.sum(self.scores['target_region_acc'])
        cur_region_iou = np.sum(self.scores['cur_region_iou'])


        nav_err = np.sum(self.scores['nav_errors'])
        oracle_error = np.sum(self.scores['oracle_errors'])
        trajectory_lengths = np.sum(self.scores['trajectory_lengths'])
        oracle_plan_errors = np.sum(self.scores['oracle_plan_errors'])

        dist_to_end_reductions = np.sum(self.scores['dist_to_end_reductions'])
        shortest_path_lengths = np.sum(self.scores['shortest_path_lengths'])

        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        oracle_plan_successes = len([i for i in self.scores['oracle_plan_errors'] if i < self.error_margin])

        spls = []
        for err, length, sp in zip(self.scores['nav_errors'], self.scores['trajectory_lengths'],
                                   self.scores['shortest_path_lengths']):
            if err < self.error_margin:
                if sp > 0:
                    spls.append(sp / max(length, sp))
                else:  # In IF, some Q/A pairs happen when we're already in the goal region, so taking no action is correct.
                    spls.append(1 if length == 0 else 0)
            else:
                spls.append(0)
        spl = sum(spls)
        task_num = len(self.scores['trajectory_lengths'])

        # cal for repated_task
        if self.repeated_score is None:
            repeated_res = [0] * 15
        else:
            r_num_succ = 1 if self.repeated_score['nav_errors'] < self.error_margin else 0
            r_oracle_succ = 1 if self.repeated_score['oracle_errors'] < self.error_margin else 0
            r_oracle_plan_succ = 1 if self.repeated_score['oracle_plan_errors'] < self.error_margin else 0
            r_spl = float(self.repeated_score['nav_errors'] < self.error_margin) * \
                    self.repeated_score['shortest_path_lengths'] / \
                    max(self.repeated_score['trajectory_lengths'], self.repeated_score['shortest_path_lengths'], 0.01)

            repeated_res = [self.repeated_score['trajectory_lengths'],
                            self.repeated_score['nav_errors'],
                            r_oracle_succ,
                            r_num_succ,
                            r_spl,
                            r_oracle_plan_succ,
                            self.repeated_score['dist_to_end_reductions'],
                            self.repeated_score['angle_acc'], self.repeated_score['next_region_acc'],
                            self.repeated_score['target_region_acc'], self.repeated_score['cur_region_iou'],
                            self.repeated_score['num_angle_pred'], self.repeated_score['num_region_pred'],
                            self.repeated_score['num_candi_region_pred'],
                            1]

        return torch.tensor([[trajectory_lengths, nav_err, oracle_successes, num_successes, spl, oracle_plan_successes,
                              dist_to_end_reductions, angle_acc, next_region_acc, target_region_acc,
                               cur_region_iou, num_angle_pred, num_region_pred, num_candi_region_pred,
                              task_num], repeated_res], dtype=torch.float).cuda()

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['inst_idx'] in instr_ids:
                instr_ids.remove(item['inst_idx'])
                self._score_item(item['inst_idx'], item['trajectory'])
        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s' \
                                        % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'vp_steps': np.average(self.scores['vp_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes) / float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes) / float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
               for error, p, l in
               zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
               ]
        score_summary['spl'] = np.average(spl)
        return score_summary, self.scores


class MultiEvaluationREVERIE(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok, args=None):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []
        self.args = args
        for split in splits:
            for item in load_datasets([split], args.task):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        self.objProposals, self.obj2viewpoint = self.loadObjProposals(args.bboxDir)

    def loadObjProposals(self, bboxDir):
        objProposals = {}
        obj2viewpoint = {}
        for efile in os.listdir(bboxDir):
            if efile.endswith('.json'):
                with open(os.path.join(bboxDir, efile), 'r') as f:
                    scan = efile.split('_')[0]
                    scanvp, _ = efile.split('.')
                    data = json.load(f)
                    for vp, vv in data.items():
                        for objid, objinfo in vv.items():
                            if objinfo['visible_pos']:
                                if obj2viewpoint.__contains__(scan+'_'+objid):
                                    if vp not in obj2viewpoint[scan+'_'+objid]:
                                        obj2viewpoint[scan+'_'+objid].append(vp)
                                else:
                                    obj2viewpoint[scan+'_'+objid] = [vp,]

                                if objProposals.__contains__(scanvp):
                                    for ii, bbox in enumerate(objinfo['bbox2d']):
                                        objProposals[scanvp]['bbox'].append(bbox)
                                        objProposals[scanvp]['visible_pos'].append(objinfo['visible_pos'][ii])
                                        objProposals[scanvp]['objId'].append(objid)

                                else:
                                    objProposals[scanvp] = {'bbox': objinfo['bbox2d'],
                                                            'visible_pos': objinfo['visible_pos']}
                                    objProposals[scanvp]['objId'] = []
                                    for _ in objinfo['visible_pos']:
                                        objProposals[scanvp]['objId'].append(objid)

        return objProposals, obj2viewpoint

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, instr_id, item_in, evalType):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        path = item_in['trajectory']

        if evalType == 'whole':
            predObjId = item_in['predObjId'] # open when release
        if self.splits == 'test':
            gt = self.gt[instr_id.split('_')[0]]
        else:
            gt = self.gt[instr_id.split('_')[0]+'_'+instr_id.split('_')[1]]

        objId = str(gt['objId'])
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]

        # correct the goal
        scan = gt['scan']
        candidate_vps = []
        for cvp in self.obj2viewpoint[scan + '_' + objId]:
            if self.distances[scan][start].__contains__(cvp):
                candidate_vps.append(cvp)

        # remote grounding success or not #open when release
        if evalType == 'whole':
            if objId==str(predObjId):
                self.scores['rgs'].append(1)
            else:
                self.scores['rgs'].append(0)

        # success or not
        if self.objProposals.__contains__(scan+'_'+path[-1][0]):
            if objId in self.objProposals[scan+'_'+path[-1][0]]['objId']:
                self.scores['visible'].append(1)
            else:
                self.scores['visible'].append(0)
        else:
            self.scores['visible'].append(0)

        # oracle success or not
        oracle_succ = 0
        for passvp in path:
            if self.objProposals.__contains__(scan+'_'+passvp[0]):
                if objId in self.objProposals[scan+'_'+passvp[0]]['objId']:
                    oracle_succ = 1
                    break
        self.scores['oracle_visible'].append(oracle_succ)

        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        vp_step = 0
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            if curr[0]!=prev[0]:
                vp_step+=1
            prev = curr
        self.scores['steps'].append(vp_step)
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_path_lengths'].append(self.distances[gt['scan']][start][goal])

        self.scores['num_angle_pred'].append(len(item_in['angle_pred']))
        self.scores['num_region_pred'].append(len(item_in['target_region_pred']))
        self.scores['num_candi_region_pred'].append(len(item_in['cur_region_pred']))


        self.scores['angle_acc'].append(sum(item_in['angle_pred']))
        self.scores['next_region_acc'].append(sum(item_in['next_region_pred']))
        self.scores['target_region_acc'].append(sum(item_in['target_region_pred']))
        self.scores['cur_region_iou'].append(sum(item_in['cur_region_pred']))

    def distributed_score(self, output_file, repeated_task_id, evalType='nav'):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        self.repeated_score = None
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                # instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item, evalType)
                if item['instr_id'] == repeated_task_id:
                    self.repeated_score = {}
                    for key in self.scores.keys():
                        self.repeated_score[key] = self.scores[key][-1]
                # if item['instr_id'] == repeated_task_id:
                #     self.repeated_score={'visible':self.scores['visible'][-1],
                #                          'oracle_visible':self.scores['oracle_visible'][-1],
                #                          'trajectory_lengths':self.scores['trajectory_lengths'][-1],
                #                          'steps': self.scores['steps'][-1],
                #                          'shortest_path_lengths': self.scores['shortest_path_lengths'][-1],
                #                          }
        # temporarily not support 'whole'
        if evalType == 'whole':
            assert len(self.scores['rgs']) == len(self.instr_ids)
            num_rgs = np.sum(self.scores['rgs'])

        # multi loss
        num_angle_pred = np.sum(self.scores['num_angle_pred'])
        num_region_pred = np.sum(self.scores['num_region_pred'])
        num_candi_region_pred = np.sum(self.scores['num_candi_region_pred'])

        angle_acc = np.sum(self.scores['angle_acc'])
        next_region_acc = np.sum(self.scores['next_region_acc'])
        target_region_acc = np.sum(self.scores['target_region_acc'])
        cur_region_iou = np.sum(self.scores['cur_region_iou'])

        num_successes = np.sum(self.scores['visible'])
        oracle_successes = np.sum(self.scores['oracle_visible'])
        steps = np.sum(self.scores['steps'])
        lengths = np.sum(self.scores['trajectory_lengths'])
        spl = []
        for visible, length, sp in zip(self.scores['visible'], self.scores['trajectory_lengths'],
                                   self.scores['shortest_path_lengths']):
            if visible:
                spl.append(sp / max(length, sp))
            else:
                spl.append(0)
        spl = np.sum(spl)

        # temporarily not support 'whole'
        if evalType == 'whole':
            wrgs = []
            for rgs, length, sp in zip(self.scores['rgs'], self.scores['trajectory_lengths'],
                                           self.scores['shortest_path_lengths']):
                if rgs:
                    wrgs.append(sp / max(length, sp))
                else:
                    wrgs.append(0)
            wrgs = np.sum(wrgs)

        task_num = len(self.scores['trajectory_lengths'])
        # cal for repated_task
        if self.repeated_score is None:
            repeated_res = [0] * 13
        else:
            r_num_succ= self.repeated_score['visible']
            r_oracle_succ = self.repeated_score['oracle_visible']
            r_step = self.repeated_score['steps']
            r_spl = float(self.repeated_score['visible']) * \
                            self.repeated_score['shortest_path_lengths'] / \
                            max(self.repeated_score['trajectory_lengths'], self.repeated_score['shortest_path_lengths'], 0.01)

            repeated_res = [self.repeated_score['trajectory_lengths'], r_num_succ, r_oracle_succ, r_spl, r_step,
                            self.repeated_score['angle_acc'], self.repeated_score['next_region_acc'],
                            self.repeated_score['target_region_acc'], self.repeated_score['cur_region_iou'],
                            self.repeated_score['num_angle_pred'], self.repeated_score['num_region_pred'],
                            self.repeated_score['num_candi_region_pred'],
                            1]

        return torch.tensor([[lengths, num_successes, oracle_successes, spl, steps, angle_acc, next_region_acc, target_region_acc,
                               cur_region_iou, num_angle_pred, num_region_pred, num_candi_region_pred, task_num],
                             repeated_res],dtype=torch.float).cuda()

    def score(self, output_file):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                instr_ids.remove(item['instr_id'])
                self._score_item(item['instr_id'], item['trajectory'])
        if 'train' not in self.splits:  # Exclude the training from this. (Because training eval may be partial)
            assert len(instr_ids) == 0, 'Missing %d of %d instruction ids from %s - not in %s'\
                           % (len(instr_ids), len(self.instr_ids), ",".join(self.splits), output_file)
            assert len(self.scores['nav_errors']) == len(self.instr_ids)
        score_summary = {
            'nav_error': np.average(self.scores['nav_errors']),
            'oracle_error': np.average(self.scores['oracle_errors']),
            'steps': np.average(self.scores['trajectory_steps']),
            'vp_steps': np.average(self.scores['vp_steps']),
            'lengths': np.average(self.scores['trajectory_lengths'])
        }
        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        score_summary['success_rate'] = float(num_successes)/float(len(self.scores['nav_errors']))
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])
        score_summary['oracle_rate'] = float(oracle_successes)/float(len(self.scores['oracle_errors']))

        spl = [float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ]
        score_summary['spl'] = np.average(spl)
        return score_summary, self.scores


class MultiEvaluation(object):
    ''' Results submission format:  [{'instr_id': string, 'trajectory':[(viewpoint_id, heading_rads, elevation_rads),] } ] '''

    def __init__(self, splits, scans, tok, args=None):
        self.error_margin = 3.0
        self.splits = splits
        self.tok = tok
        self.gt = {}
        self.instr_ids = []
        self.scans = []

        self.args = args
        for split in splits:
            for item in load_datasets([split], args.task):
                if scans is not None and item['scan'] not in scans:
                    continue
                self.gt[str(item['path_id'])] = item
                self.scans.append(item['scan'])
                self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]
        self.scans = set(self.scans)
        self.instr_ids = set(self.instr_ids)
        self.graphs = load_nav_graphs(self.scans)
        self.distances = {}
        for scan,G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _get_nearest(self, scan, goal_id, path):
        near_id = path[0][0]
        near_d = self.distances[scan][near_id][goal_id]
        for item in path:
            d = self.distances[scan][item[0]][goal_id]
            if d < near_d:
                near_id = item[0]
                near_d = d
        return near_id

    def _score_item(self, item):
        ''' Calculate error based on the final position in trajectory, and also
            the closest position (oracle stopping rule).
            The path contains [view_id, angle, vofv] '''
        instr_id, path = item['instr_id'], item['trajectory']
        gt = self.gt[instr_id.split('_')[-2]]
        start = gt['path'][0]
        assert start == path[0][0], 'Result trajectories should include the start position'
        goal = gt['path'][-1]
        final_position = path[-1][0]    # the first of [view_id, angle, vofv]
        nearest_position = self._get_nearest(gt['scan'], goal, path)
        self.scores['nav_errors'].append(self.distances[gt['scan']][final_position][goal])
        self.scores['oracle_errors'].append(self.distances[gt['scan']][nearest_position][goal])
        self.scores['trajectory_steps'].append(len(path)-1)
        distance = 0 # Work out the length of the path in meters
        prev = path[0]
        vp_step=0
        for curr in path[1:]:
            distance += self.distances[gt['scan']][prev[0]][curr[0]]
            if curr[0]!=prev[0]:
                vp_step+=1
            prev = curr
        self.scores['vp_steps'].append(vp_step)
        self.scores['trajectory_lengths'].append(distance)
        self.scores['shortest_lengths'].append(
            self.distances[gt['scan']][start][goal]
        )
        self.scores['num_angle_pred'].append(len(item['angle_pred']))
        self.scores['num_region_pred'].append(len(item['target_region_pred']))
        self.scores['num_candi_region_pred'].append(len(item['cur_region_pred']))


        self.scores['angle_acc'].append(sum(item['angle_pred']))
        self.scores['next_region_acc'].append(sum(item['next_region_pred']))
        self.scores['target_region_acc'].append(sum(item['target_region_pred']))
        self.scores['cur_region_iou'].append(sum(item['cur_region_pred']))


    def distributed_score(self, output_file, repeated_task_id):
        ''' Evaluate each agent trajectory based on how close it got to the goal location '''
        self.scores = defaultdict(list)
        self.repeated_score = None
        instr_ids = set(self.instr_ids)
        if type(output_file) is str:
            with open(output_file) as f:
                results = json.load(f)
        else:
            results = output_file
        for item in results:
            # Check against expected ids
            if item['instr_id'] in instr_ids:
                # instr_ids.remove(item['instr_id'])
                self._score_item(item)
                if item['instr_id'] == repeated_task_id:
                    self.repeated_score = {}
                    for key in self.scores.keys():
                        self.repeated_score[key] = self.scores[key][-1]

        nav_err = np.sum(self.scores['nav_errors'])
        oracle_error =  np.sum(self.scores['oracle_errors'])
        steps = np.sum(self.scores['trajectory_steps'])
        vp_steps = np.sum(self.scores['vp_steps'])
        lengths = np.sum(self.scores['trajectory_lengths'])

        num_angle_pred = np.sum(self.scores['num_angle_pred'])
        num_region_pred = np.sum(self.scores['num_region_pred'])
        num_candi_region_pred = np.sum(self.scores['num_candi_region_pred'])

        angle_acc = np.sum(self.scores['angle_acc'])
        next_region_acc = np.sum(self.scores['next_region_acc'])
        target_region_acc = np.sum(self.scores['target_region_acc'])
        cur_region_iou = np.sum(self.scores['cur_region_iou'])

        num_successes = len([i for i in self.scores['nav_errors'] if i < self.error_margin])
        oracle_successes = len([i for i in self.scores['oracle_errors'] if i < self.error_margin])

        spl = np.sum([float(error < self.error_margin) * l / max(l, p, 0.01)
            for error, p, l in
            zip(self.scores['nav_errors'], self.scores['trajectory_lengths'], self.scores['shortest_lengths'])
        ])
        task_num = len(self.scores['trajectory_steps'])
        # cal for repated_task
        if self.repeated_score is None:
            repeated_res = [0] * 16
        else:
            r_num_succ= 1 if self.repeated_score['nav_errors'] < self.error_margin else 0
            r_oracle_succ = 1 if self.repeated_score['oracle_errors'] < self.error_margin else 0

            r_spl = float(self.repeated_score['nav_errors'] < self.error_margin) * \
                            self.repeated_score['shortest_lengths'] / \
                            max(self.repeated_score['trajectory_lengths'], self.repeated_score['shortest_lengths'], 0.01)

            repeated_res = [self.repeated_score['nav_errors'], self.repeated_score['oracle_errors'],
                            self.repeated_score['trajectory_steps'], self.repeated_score['vp_steps'],
                            self.repeated_score['trajectory_lengths'],
                            r_num_succ, r_oracle_succ, r_spl,
                            self.repeated_score['angle_acc'], self.repeated_score['next_region_acc'],
                            self.repeated_score['target_region_acc'], self.repeated_score['cur_region_iou'],
                            self.repeated_score['num_angle_pred'],self.repeated_score['num_region_pred'],
                            self.repeated_score['num_candi_region_pred'],
                            1]

        return  torch.tensor([[nav_err, oracle_error, steps, vp_steps, lengths,num_successes,
                        oracle_successes, spl, angle_acc, next_region_acc, target_region_acc,
                cur_region_iou, num_angle_pred, num_region_pred, num_candi_region_pred, task_num],repeated_res],dtype=torch.float).cuda()


RESULT_DIR = 'tasks/R2R/results/'

