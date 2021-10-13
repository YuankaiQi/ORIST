"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

saving utilities
"""
import json
import os
from os.path import abspath, dirname, exists, join
import subprocess

import torch

from utils.logger import LOGGER


def save_training_meta(args):
    if args.local_rank > 0:
        return

    if not exists(join(args.output_dir, 'log')):
        os.makedirs(join(args.output_dir, 'log'))
    if not exists(join(args.output_dir, 'ckpt')):
        os.makedirs(join(args.output_dir, 'ckpt'))

    with open(join(args.output_dir, 'log', 'args.json'), 'w') as writer:
        json.dump(vars(args), writer, indent=4)
    model_config = json.load(open(args.model_config))
    with open(join(args.output_dir, 'log', 'model.json'), 'w') as writer:
        json.dump(model_config, writer, indent=4)
    # git info


class ModelSaver(object):
    def __init__(self, output_dir, prefix='model_step', suffix='pt'):
        self.output_dir = output_dir
        self.prefix = prefix
        self.suffix = suffix

    def save(self, model, step, optimizer, amp):
        output_model_file = join(self.output_dir,
                                 f"{self.prefix}_{step}.{self.suffix}")
        state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in model.state_dict().items()}

        torch.save(state_dict, output_model_file)

        ouput_state_file = f'{self.output_dir}/train_state_{step}.pt'
        torch.save({
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'global_step':step
                    }, ouput_state_file)

        return output_model_file, ouput_state_file
