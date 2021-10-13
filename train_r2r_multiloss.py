#python -m torch.distributed.launch --nproc_per_node=
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn

from data import R2RDataset
from utils.logger import LOGGER, TB_LOGGER, add_log_to_file

from utils.save import save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
import os
from os.path import abspath, dirname, exists, join
from time import time
import json
import math
import pdb
from optim import get_lr_sched
import numpy as np
from collections import defaultdict

from r2r_utils import (read_vocab,write_vocab,build_vocab,Tokenizer,VLNBertTokenizer,timeSince,
                       read_img_features,read_img_features_h5,create_links,reduce_tensor)

from env import R2RBatch
from agent_r2r_multiloss import BertAgent
from eval import MultiEvaluation
from r2r_param_multiloss import args
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from tensorboardX import SummaryWriter
from speaker import Speaker
metrics = ['nav_error','oracle_error','steps','vp_steps','lengths','success_rate','oracle_rate',
           'spl', 'angle_acc', 'next_region_acc', 'target_region_acc', 'candi_region_acc']
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
args.output_dir = 'snap/%s_%s_%s/%s' % (args.project, args.task, args.name, TIMESTAMP)
args.log_dir = os.path.join(args.output_dir, 'log')
print('Log dir = %s'%args.log_dir)

if args.local_rank == 0:
    create_links(args)
    TB_LOGGER.create(args.log_dir)
    # pbar = tqdm(total=args.num_train_steps)
    add_log_to_file(join(args.log_dir, 'log.txt'))
else:
    LOGGER.disabled = True

LOGGER.info('PyTorch Version = '+ str(torch.__version__))
LOGGER.info('PyTorch Cuda Viersion = ' + str(torch.version.cuda))
LOGGER.info('PyTorch Cuda Available = ' + str(torch.cuda.is_available()))

feedback_method = args.feedback # teacher or sample

def wrt_log(listner, writer, interval, idx):
    # Log the training stats to tensorboard
    ml_action_num = max(listner.logs['ml_action_num'], 1)
    rl_action_num = max(listner.logs['rl_action_num'], 1)
    ml_loss = listner.logs['ml_loss'] / ml_action_num
    rl_loss = listner.logs['rl_loss'] / rl_action_num
    ml_step = listner.logs['ml_step'] / max(interval, 1)
    rl_step = listner.logs['rl_step'] / max(interval, 1)
    all_step = ml_step + rl_step

    # multiloss
    nav_loss = listner.logs['nav_loss'] / ml_action_num
    critic_loss = listner.logs['critic_loss'] / rl_action_num  # / length / args.batchSize
    entropy = listner.logs['entropy'] / rl_action_num  # / length / args.batchSize

    writer.add_scalar("loss/critic", critic_loss, idx)
    writer.add_scalar("loss/ml", ml_loss, idx)
    writer.add_scalar("loss/rl", rl_loss, idx)
    writer.add_scalar("loss/nav_loss", nav_loss, idx)
    writer.add_scalar("policy_entropy", entropy, idx)
    writer.add_scalar("steps/ml", ml_step, idx)
    writer.add_scalar("steps/rl", rl_step, idx)
    writer.add_scalar("steps/all", all_step, idx)

    if args.angle_loss:
        angle_loss = listner.logs['angle_loss'] / (ml_action_num + rl_action_num)
        writer.add_scalar("loss/angle_loss", angle_loss, idx)

    if args.progress_loss:
        progress_loss = listner.logs['progress_loss'] / ml_action_num
        writer.add_scalar("loss/progress_loss", progress_loss, idx)

    if args.candi_region_loss:
        region_loss = listner.logs['region_loss'] / (ml_action_num + rl_action_num)
        writer.add_scalar("loss/candi_region_loss", region_loss, idx)

    if args.next_region_loss:
        next_region_loss = listner.logs['next_region_loss'] / (ml_action_num + rl_action_num)
        writer.add_scalar("loss/next_region_loss", next_region_loss, idx)

    if args.target_region_loss:
        target_region_loss = listner.logs['target_region_loss'] / (ml_action_num + rl_action_num)
        writer.add_scalar("loss/target_region_loss", target_region_loss, idx)

    if args.speaker_loss:
        speaker_loss = listner.logs['speaker_loss'] / (ml_action_num + rl_action_num)
        writer.add_scalar("loss/speaker_loss", speaker_loss, idx)

def valid(train_env, tok, val_envs={}):
    # writer = SummaryWriter(logdir=log_dir)
    listner = BertAgent(train_env, "", tok, args.maxAction, args)

    best_val = {'val_seen': {"accu": 0., "state": "", 'update': False},
                'val_unseen': {"accu": 0., "state": "", 'update': False}}
    listner.best_val = best_val
    listner.logs = defaultdict(float)
    log_keys = ['ml_loss', 'rl_loss', 'ml_action_num',
                'rl_action_num', 'entropy', 'critic_loss',
                'ml_step', 'rl_step', 'eval_ml_loss', 'eval_action_num']
    for log_k in log_keys:
        listner.logs[log_k] = 0

    start_epoch = listner.start_epoch

    val_loaders, repeated_task = {},{}

    if args.local_rank == 0:
        save_training_meta(args)
        print(args)
    for env_name, (env, evaluator) in val_envs.items():
        val_split_data = R2RDataset(len(env.data), env_name, args.n_gpu, args.val_batch_size)
        if env_name == 'train':
            repeated_task[env_name] = {'repeated_id': 'None',
                                       'extra_num': 0}  # repeat the last task
        else:
            repeated_task[env_name] = {'repeated_id':env.index2instr_id[len(env.data) - 1],
                                   'extra_num':val_split_data.get_extra_data_num()} # repeat the last task
        val_sampler = DistributedSampler(val_split_data)
        val_loader = DataLoader(dataset=val_split_data, batch_size=args.val_batch_size, shuffle=False, sampler=val_sampler)
        val_loaders[env_name]=val_loader

    LOGGER.info(f"***** Running training with {args.n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", args.train_batch_size * args.n_gpu)
    LOGGER.info("  Accumulate steps = %d", args.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", args.num_train_steps)

    loss_str = {}
    listner.phase = 'test'
    listner.model.eval()
    listner.feedback = 'argmax'
    listner.args.submit = True
    for env_name, (env, evaluator) in val_envs.items():
        LOGGER.info('Eval %s' % env_name)
        LOGGER.info("Task Num = %d"%val_loaders[env_name].dataset.len)
        listner.env = env
        listner.logs['eval_action_num'] = 0
        result = []
        loss_str[env_name] = "\t"
        # Get validation loss under the same conditions as training
        ids_num = 0
        if env_name == 'train':
            # train_batch_ids = []
            for step, batch_ids in enumerate(val_loaders[env_name]):
                if args.local_rank == 0:
                    ids_num += len(batch_ids)
                    print(ids_num)
                listner.env.prepare_batch(batch_ids)
                for traj in listner.rollout():
                    traj['trajectory'] = traj['path']
                    del traj['path']
                    result.append(traj)
                    # result.append({'instr_id': traj['instr_id'], 'trajectory': traj['path']})
                if args.debug:
                    if step == 2:
                        break
                elif step == math.ceil(1280 / (args.val_batch_size * args.n_gpu)):
                    break

        else:
            # indices=[]
            for step, batch_ids in enumerate(val_loaders[env_name]):
                if args.local_rank == 0:
                    ids_num += len(batch_ids)
                    print(ids_num)
                listner.env.prepare_batch(batch_ids)
                # indices.append(batch_ids)
                # trajectories = listner.rollout()
                for traj in listner.rollout():
                    traj['trajectory'] = traj['path']
                    del traj['path']
                    result.append(traj)

        torch.distributed.barrier()
        if args.local_rank == 0:
            LOGGER.info("Rank 0, Result Len = %d" % len(result))

            # remove repeat task
            new_res = []
            instr_ids = []
            for item in result:
                if item['instr_id'] not in instr_ids:
                    instr_ids.append(item['instr_id'])
                    new_item = {}
                    new_item['instr_id'] = item['instr_id']
                    new_item['trajectory'] = item['trajectory']
                    new_res.append(new_item)
            # remake result dir
            result_dir = args.resume + '_eval_res'
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            result_path = os.path.join(result_dir, '%s_submit.json') % env_name
            with open(result_path, 'w+') as f:
                json.dump(new_res, f, indent=2)
                LOGGER.info("Save Result : %s" % str(result_path))

        score_summary = evaluator.distributed_score(result, repeated_task[env_name]['repeated_id'])
        rt = reduce_tensor(score_summary, dst=0)

        if env_name != 'test':
            if args.local_rank == 0:
                if repeated_task[env_name]['extra_num'] != 0:
                    gg = rt[0, :] - rt[1, :] / rt[1, -1] * repeated_task[env_name]['extra_num']
                else:
                    gg = rt[0, :]
                num_angle_pred, num_region_pred, num_candi_region_pred, task_num = gg[-4], gg[-3], gg[-2], gg[-1]
                # gg: nav_err, oracle_error, steps, lengths, num_successes,oracle_successes,spl,task_num
                loss_str[env_name] += "%s result: " % env_name
                for i, metric in enumerate(metrics):
                    if not args.angle_loss and 'angle_acc' == metric:
                        continue
                    if not args.next_region_loss and 'next_region_acc' == metric:
                        continue
                    if not args.target_region_loss and 'target_region_acc' == metric:
                        continue
                    if not args.candi_region_loss and 'candi_region_acc' == metric:
                        continue

                    if metric == 'angle_acc':
                        val = gg[i] / num_angle_pred
                    elif metric in ['next_region_acc', 'target_region_acc']:
                        val = gg[i] / num_region_pred
                    elif metric == 'candi_region_acc':
                        val = gg[i] / num_candi_region_pred
                    else:
                        val = gg[i] / task_num
                    loss_str[env_name] += '%s: %.3f,' % (metric, val)
                    if metric in ['success_rate']:
                        if env_name in best_val:
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True

    if args.local_rank == 0:
        long_str = []
        for env_name, (_, _) in val_envs.items():
            LOGGER.info(loss_str[env_name])
            long_str.append(loss_str[env_name])

        for env_name in best_val:
            if best_val[env_name]['update']:
                best_val[env_name]['state'] = ['  %s ' % (env_name)] + long_str
                best_val[env_name]['update'] = False
                listner.best_val = best_val

        LOGGER.info("*****BEST RESULT TILL NOW")
        for env_name in best_val:
            for linfo in best_val[env_name]['state']:
                LOGGER.info(linfo)


def train(train_env, tok, val_envs={}, aug_env=None):
    if aug_env is None:
        listner = BertAgent(train_env, "", tok, args.maxAction, args)
    else:
        listner = BertAgent(aug_env, "", tok, args.maxAction, args)

    speaker = None

    if args.self_train:
        speaker = Speaker(train_env, listner, tok)
        if args.speaker is not None:
            print("Load the speaker from %s" % args.speaker)
            speaker.load(args.speaker)

    if listner.resumed:
        best_val = listner.best_val
        LOGGER.info('start from epoch %d, global_step %d'%(listner.start_epoch,listner.global_step))
    else:
        best_val = {'val_seen': {"accu": 0., "state":"", 'update':False},
                'val_unseen': {"accu": 0., "state":"", 'update':False}}
        listner.best_val = best_val
        listner.logs = defaultdict(float)
        log_keys = ['ml_loss', 'rl_loss', 'ml_action_num',
                    'rl_action_num', 'entropy', 'critic_loss',
                    'ml_step', 'rl_step', 'eval_ml_loss', 'eval_action_num']
        for log_k in log_keys:
            listner.logs[log_k] = 0

    if args.log_every == -1:
        log_every = 20 #math.ceil(len(train_env.data) / args.batchSize)
    else:
        log_every = args.log_every

    start_epoch = listner.start_epoch

    if args.debug:
        LOGGER.info('You are in debug mode!')
        log_every = 1
        args.num_train_epochs = start_epoch+4

    previous_model=''

    train_split = R2RDataset(len(train_env.data),'train', args.n_gpu, args.train_batch_size)
    train_sampler = DistributedSampler(train_split)
    train_loader = DataLoader(dataset=train_split, batch_size=args.train_batch_size, shuffle=False, sampler=train_sampler)
    val_loaders, repeated_task = {},{}

    if aug_env:
        aug_split = R2RDataset(len(aug_env.data), 'train', args.n_gpu, args.train_batch_size)
        aug_sampler = DistributedSampler(aug_split)
        aug_loader = DataLoader(dataset=aug_split, batch_size=args.train_batch_size, shuffle=False, sampler=aug_sampler)

    if args.local_rank == 0:
        save_training_meta(args)
        print(args)
    for env_name, (env, evaluator) in val_envs.items():
        val_split_data = R2RDataset(len(env.data), env_name, args.n_gpu, args.val_batch_size)
        if env_name == 'train':
            repeated_task[env_name] = {'repeated_id': 'None',
                                       'extra_num': 0}  # repeat the last task
        else:
            repeated_task[env_name] = {'repeated_id':env.index2instr_id[len(env.data) - 1],
                                   'extra_num':val_split_data.get_extra_data_num()} # repeat the last task
        val_sampler = DistributedSampler(val_split_data)
        val_loader = DataLoader(dataset=val_split_data, batch_size=args.val_batch_size, shuffle=False, sampler=val_sampler)
        val_loaders[env_name]=val_loader

    LOGGER.info(f"***** Running training with {args.n_gpu} GPUs *****")
    LOGGER.info("  Batch size = %d", args.train_batch_size * args.n_gpu)
    LOGGER.info("  Accumulate steps = %d", args.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", args.num_train_steps)


    start = time()
    start_step = listner.global_step
    small_train_epoch = -1

    for epochId in range(start_epoch, args.num_train_epochs):

        train_sampler.set_epoch(epochId)
        listner.phase = 'train'
        listner.args.submit = False
        listner.model.train()

        if aug_env is None:  # The default training process
            listner.env = train_env

            for step, batch_ids in enumerate(train_loader):
                if args.debug:
                    if step == 10:
                        break
                # n_examples += torch.tensor(len(batch_ids)).cuda()
                listner.global_step += 1
                TB_LOGGER.step()

                listner.optimizer.zero_grad()
                listner.loss = 0
                listner.train_with_grad_accumulate(1, feedback=feedback_method, batch_ids=batch_ids)  # Train interval iters
                listner.loss.backward()
                # set learning rate
                lr_this_step = get_lr_sched(listner.global_step, args)
                for i, param_group in enumerate(listner.optimizer.param_groups):
                    if i < len(listner.optimizer.param_groups) - 2:  # orien, head, critic
                        param_group['lr'] = lr_this_step * args.lr_mul
                    else:  # backbone
                        param_group['lr'] = lr_this_step

                # update params
                if args.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(listner.model.parameters(),
                                                               args.grad_norm)
                listner.optimizer.step()  # call this before lr_scheduler if use pytorch 1.1 and later


                if listner.global_step%log_every == 0:
                    # log loss
                    # NOTE: not gathered across GPUs for efficiency
                    TB_LOGGER.add_scalar('lr', lr_this_step, listner.global_step)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, listner.global_step)
                    wrt_log(listner, TB_LOGGER, log_every, listner.global_step)
                    listner.logs['ml_step'] = 0 # batchsize independent
                    listner.logs['rl_step'] = 0
                if args.debug:
                    LOGGER.info(f'============Step {listner.global_step}=============')
                    ex_per_sec = ( (listner.global_step-start_step)/(time() - start))
                    remain_hour = (args.num_train_steps-listner.global_step)/ex_per_sec/(60*60)
                    LOGGER.info(f'{ex_per_sec} step/s, remaining {remain_hour} hours')
        else:
            aug_sampler.set_epoch(epochId)
            small_train_epoch += 1
            train_sampler.set_epoch(small_train_epoch)
            train_iterator = iter(train_loader)
            for step, aug_batch_ids in enumerate(aug_loader):

                if args.debug and step >= 10:
                    break
                try:
                    batch_ids = next(train_iterator)
                except StopIteration:
                    small_train_epoch += 1
                    train_sampler.set_epoch(small_train_epoch)
                    train_iterator = iter(train_loader)
                    batch_ids = next(train_iterator)

                listner.global_step += 1
                TB_LOGGER.step()
                listner.loss = 0
                listner.optimizer.zero_grad()
                listner.env = train_env

                # Train with GT data
                args.ml_weight = 0.2
                listner.train_with_grad_accumulate(1, feedback=feedback_method, batch_ids=batch_ids)
                listner.loss.backward()

                lr_this_step = get_lr_sched(listner.global_step, args)
                for i, param_group in enumerate(listner.optimizer.param_groups):
                    if i < len(listner.optimizer.param_groups) - 2:  # orien, head, critic
                        param_group['lr'] = lr_this_step * args.lr_mul
                    else:  # backbone
                        param_group['lr'] = lr_this_step

                # update params
                if args.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(listner.model.parameters(),
                                                               args.grad_norm)
                listner.optimizer.step()

                # Train with Back Translation
                listner.loss = 0
                listner.optimizer.zero_grad()
                listner.env = aug_env
                args.ml_weight = 0.2  # Sem-Configuration
                listner.train_with_grad_accumulate(1, feedback=feedback_method, batch_ids=aug_batch_ids, speaker=speaker)
                listner.loss.backward()
                # set learning rate
                lr_this_step = get_lr_sched(listner.global_step, args)
                for i, param_group in enumerate(listner.optimizer.param_groups):
                    if i < len(listner.optimizer.param_groups) - 2:  # orien, head, critic
                        param_group['lr'] = lr_this_step * args.lr_mul
                    else:  # backbone
                        param_group['lr'] = lr_this_step

                # update params
                if args.grad_norm != -1:
                    grad_norm = torch.nn.utils.clip_grad_norm_(listner.model.parameters(),
                                                               args.grad_norm)
                listner.optimizer.step()

                if listner.global_step % log_every == 0:
                    # log loss
                    TB_LOGGER.add_scalar('lr', lr_this_step, listner.global_step)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, listner.global_step)
                    wrt_log(listner, TB_LOGGER, log_every, listner.global_step)
                    listner.logs['ml_step'] = 0  # batchsize independent
                    listner.logs['rl_step'] = 0
                if args.debug:
                    LOGGER.info(f'============Step {listner.global_step}=============')
                    ex_per_sec = ((listner.global_step - start_step) / (time() - start))
                    remain_hour = (args.num_train_steps - listner.global_step) / ex_per_sec / (60 * 60)
                    LOGGER.info(f'{ex_per_sec} step/s, remaining {remain_hour} hours')

        loss_str = {}
        listner.phase = 'test'
        listner.model.eval()
        listner.feedback = 'argmax'
        listner.args.submit=True
        for env_name, (env, evaluator) in val_envs.items():
            listner.env = env
            listner.logs['eval_action_num'] = 0
            result = []
            loss_str[env_name]="\t"
            # Get validation loss under the same conditions as training
            if env_name == 'train':
                val_loaders[env_name].sampler.set_epoch(epochId)
                # train_batch_ids = []
                for step, batch_ids in enumerate(val_loaders[env_name]):
                    # train_batch_ids += batch_ids.cpu().clone().tolist()

                    listner.env.prepare_batch(batch_ids)
                    for traj in listner.rollout():
                        traj['trajectory'] = traj['path']
                        del traj['path']
                        result.append(traj)
                        # result.append({'instr_id': traj['instr_id'], 'trajectory': traj['path']})
                    if args.debug:
                        if step == 2:
                            break
                    elif step == math.ceil(1280 / (args.val_batch_size * args.n_gpu)):
                        break

            else:
                # indices=[]
                for step, batch_ids in enumerate(val_loaders[env_name]):
                    listner.env.prepare_batch(batch_ids)
                    for traj in listner.rollout():
                        traj['trajectory'] = traj['path']
                        del traj['path']
                        result.append(traj)

            score_summary = evaluator.distributed_score(result, repeated_task[env_name]['repeated_id'])
            rt = reduce_tensor(score_summary, dst=0)


            if args.local_rank == 0:
                if repeated_task[env_name]['extra_num'] != 0:
                    gg = rt[0, :] - rt[1, :] / rt[1, -1] * repeated_task[env_name]['extra_num']
                else:
                    gg = rt[0, :]
                num_angle_pred, num_region_pred, num_candi_region_pred, task_num = gg[-4], gg[-3], gg[-2], gg[-1]
                # gg: nav_err, oracle_error, steps, lengths, num_successes,oracle_successes,spl,task_num
                loss_str[env_name] += "%s result: " % env_name
                for i, metric in enumerate(metrics):
                    if not args.angle_loss and 'angle_acc' == metric:
                        continue
                    if not args.next_region_loss and 'next_region_acc' == metric:
                        continue
                    if not args.target_region_loss and 'target_region_acc' == metric:
                        continue
                    if not args.candi_region_loss and 'candi_region_acc' == metric:
                        continue

                    if metric == 'angle_acc':
                        val = gg[i] / num_angle_pred
                    elif metric in ['next_region_acc', 'target_region_acc']:
                        val = gg[i] / num_region_pred
                    elif metric == 'candi_region_acc':
                        val = gg[i] / num_candi_region_pred
                    else:
                        val = gg[i] / task_num
                    loss_str[env_name] += '%s: %.3f,' % (metric, val)
                    if metric in ['success_rate']:
                        TB_LOGGER.add_scalar("SuccRate/%s" % env_name, val, epochId)
                        if env_name in best_val:
                            if val > best_val[env_name]['accu']:
                                best_val[env_name]['accu'] = val
                                best_val[env_name]['update'] = True
                    elif metric in ['spl', 'steps', 'vp_steps', 'angle_acc', 'next_region_acc', 'target_region_acc', 'candi_region_acc']:
                        TB_LOGGER.add_scalar("%s/%s" % (metric, env_name), val, epochId)

            torch.distributed.barrier()

        if args.local_rank == 0:
            LOGGER.info('*****%s (epoch %d %d%%)' % (timeSince(start, start_epoch, epochId + 1, args.num_train_epochs),
                                                      epochId, float(epochId) / args.num_train_epochs * 100))
            long_str = []
            for env_name, (_, _) in val_envs.items():
                LOGGER.info(loss_str[env_name])
                long_str.append(loss_str[env_name])

            for env_name in best_val:
                if best_val[env_name]['update']:
                    best_val[env_name]['state'] = ['  %s epochId %d:' %(env_name, epochId)] + long_str
                    best_val[env_name]['update'] = False
                    listner.best_val = best_val
                    listner.save(epochId, os.path.join(args.output_dir, 'ckpt', "best_%s" % (env_name)))

            LOGGER.info("*****BEST RESULT TILL NOW")
            for env_name in best_val:
                for linfo in best_val[env_name]['state']:
                    LOGGER.info(linfo)

            latest_model = os.path.join(args.output_dir, 'ckpt', "Iter_%04d" % (epochId))
            listner.save(epochId, latest_model)

            if os.path.exists(previous_model):
                os.remove(previous_model)
            previous_model = latest_model
        torch.distributed.barrier()

def setup():
    # Check for vocabs
    if args.local_rank==0:
        write_vocab(build_vocab(splits=['train'],task=args.task), args.TRAIN_VOCAB)
        write_vocab(build_vocab(splits=['train','val_seen','val_unseen'],task=args.task), args.TRAINVAL_VOCAB)


def train_val():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    torch.distributed.barrier()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(args.TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    bert_tok = VLNBertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    bert_tok.add_tokens(vocab, do_lower_case=True)
    torch.distributed.barrier()  # wait master process

    LOGGER.info('Vocab Size = %d' %len(bert_tok.vocab))
    LOGGER.info('use_angle_pi = %s'%str(args.use_angle_pi))
    LOGGER.info('use_lstm = %s' % str(args.use_lstm))
    LOGGER.info('add_whole_img_feat = %s' % str(args.add_whole_img_feat))
    LOGGER.info('max_bb = %s' % str(args.max_bb))

    LOGGER.info('use_speaker = %s' % str(args.speaker_loss))
    LOGGER.info('progress_loss = %s' % str(args.progress_loss))
    LOGGER.info('angle_loss = %s' % str(args.angle_loss))
    LOGGER.info('candi_region_loss = %s' % str(args.candi_region_loss))
    LOGGER.info('next_region_loss = %s' % str(args.next_region_loss))
    LOGGER.info('target_region_loss = %s' % str(args.target_region_loss))

    if args.local_rank==0:
        with open(args.model_config,'r') as f:
            cdata = json.load(f)
            cdata['use_speaker'] = args.speaker_loss
            cdata['use_lstm'] = args.use_lstm
            cdata['vocab_size'] = len(bert_tok.vocab)
            cdata['angleFeatSize'] = args.angle_feat_size
        with open(args.model_config, 'w') as f:
            json.dump(cdata,f,indent=1)
    torch.distributed.barrier()
    if args.features=='none':
        feat_dict = None
        featurized_scans = None
    else:
        feat_dict = read_img_features_h5(args.IMAGENET_FEATURES, args)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    if feedback_method == 'sample' and args.ml_weight != 0:
        train_env = R2RBatch(feat_dict, batch_size=int(args.train_batch_size / 2), splits=['train'], tokenizer=tok,
                             seed=args.seed,
                             bert_tokenizer=bert_tok, bert_padding_index=0, args=args)
    else:
        train_env = R2RBatch(feat_dict, batch_size=args.train_batch_size, splits=['train'], tokenizer=tok,
                             seed=args.seed, bert_tokenizer=bert_tok, bert_padding_index=0, args=args)

    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass

    if not args.eval_only and not args.beam:
        val_env_names.append("train")
    if args.debug:
        val_env_names=['val_seen','train'] #for fast debug

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.val_batch_size, splits=[split], tokenizer=tok,seed=args.seed, bert_tokenizer=bert_tok,
                         bert_padding_index=0, args =args),
           MultiEvaluation([split], featurized_scans, tok, args=args))
          )
         for split in val_env_names
         )
    )

    if args.train == 'listener':
        train(train_env, tok, val_envs=val_envs)
    elif args.train == 'validlistener':
        if args.beam:
            beam_valid(train_env, tok, val_envs=val_envs)
        else:
            valid(train_env, tok, val_envs=val_envs)
    elif args.train == 'speaker':
        train_speaker(train_env, tok, args.iters, val_envs=val_envs)
    elif args.train == 'validspeaker':
        valid_speaker(tok, val_envs)
    else:
        assert False

def train_val_augment():
    ''' Train on the training set, and validate on seen and unseen splits. '''
    # args.fast_train = True
    setup()
    torch.distributed.barrier()
    # Create a batch training environment that will also preprocess text
    vocab = read_vocab(args.TRAIN_VOCAB)
    tok = Tokenizer(vocab=vocab, encoding_length=args.maxInput)
    bert_tok = VLNBertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    bert_tok.add_tokens(vocab, do_lower_case=True)
    torch.distributed.barrier()  # wait master process

    LOGGER.info('Vocab Size = %d' %len(bert_tok.vocab))
    LOGGER.info('use_angle_pi = %s'%str(args.use_angle_pi))
    LOGGER.info('use_lstm = %s' % str(args.use_lstm))
    LOGGER.info('add_whole_img_feat = %s' % str(args.add_whole_img_feat))
    LOGGER.info('max_bb = %s' % str(args.max_bb))

    LOGGER.info('use_speaker = %s' % str(args.speaker_loss))
    LOGGER.info('progress_loss = %s' % str(args.progress_loss))
    LOGGER.info('angle_loss = %s' % str(args.angle_loss))
    LOGGER.info('candi_region_loss = %s' % str(args.candi_region_loss))
    LOGGER.info('next_region_loss = %s' % str(args.next_region_loss))
    LOGGER.info('target_region_loss = %s' % str(args.target_region_loss))

    if args.local_rank==0:
        with open(args.model_config,'r') as f:
            cdata = json.load(f)
            cdata['use_speaker'] = args.speaker_loss
            cdata['use_lstm'] = args.use_lstm
            cdata['vocab_size'] = len(bert_tok.vocab)
            cdata['angleFeatSize'] = args.angle_feat_size
        with open(args.model_config, 'w') as f:
            json.dump(cdata,f,indent=1)
    torch.distributed.barrier()
    if args.features=='none':
        feat_dict = None
        featurized_scans = None
    else:
        feat_dict = read_img_features_h5(args.IMAGENET_FEATURES, args)
        featurized_scans = set([key.split("_")[0] for key in list(feat_dict.keys())])

    train_batch_size = args.train_batch_size
    if feedback_method == 'sample' and args.ml_weight:
        train_batch_size = int(args.train_batch_size/2)

    train_env = R2RBatch(feat_dict, batch_size=train_batch_size, splits=['train'], tokenizer=tok,
                         seed=args.seed,
                         bert_tokenizer=bert_tok, bert_padding_index=0, args=args)

    aug_path = args.aug
    aug_env = R2RBatch(feat_dict,
                           batch_size=train_batch_size,
                           splits=[aug_path],
                           tokenizer=tok,
                           name='aug',
                           seed=args.seed,
                           bert_tokenizer=bert_tok,
                           bert_padding_index=0,
                           args=args)

    from collections import OrderedDict

    val_env_names = ['val_unseen', 'val_seen']
    if args.submit:
        val_env_names.append('test')
    else:
        pass
        # val_env_names.append('train')

    if not args.beam:
        val_env_names.append("train")
    if args.debug:
        val_env_names=['val_seen','train'] #for fast debug

    val_envs = OrderedDict(
        ((split,
          (R2RBatch(feat_dict, batch_size=args.val_batch_size, splits=[split], tokenizer=tok, seed=args.seed, bert_tokenizer=bert_tok,
                         bert_padding_index=0, args =args),
           MultiEvaluation([split], featurized_scans, tok, args=args))
          )
         for split in val_env_names
         )
    )

    train(train_env, tok, val_envs=val_envs, aug_env=aug_env)


if __name__ == "__main__":
    set_random_seed(args.seed)
    torch.cuda.set_device(args.local_rank) # prepare for .cuda()
    device = torch.device("cuda", args.local_rank)

    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    assert args.local_rank == torch.distributed.get_rank()

    n_gpu = torch.cuda.device_count()
    args.world_size = torch.distributed.get_world_size() # process num
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(device, args.n_gpu, args.local_rank, args.fp16))
    if args.feedback=='sample':
        args.train_batch_size *=2
        args.num_train_epochs *=2


    if args.train in ['speaker', 'rlspeaker', 'validspeaker',
                      'listener', 'validlistener']:
        train_val()
    elif args.train == 'auglistener':
        train_val_augment()
    else:
        assert False

    torch.distributed.destroy_process_group()

