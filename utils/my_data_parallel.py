# yuankai.qi @ 2020.07
from torch.nn.parallel import DataParallel
import torch
import traceback
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel.parallel_apply import parallel_apply

def scatter(inputs, target_gpus, chunk_sizes, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            try:
                return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
            except Exception as e:
                traceback.print_exc()
                print('obj', obj.size())
                print('dim', dim)
                print('chunk_sizes', chunk_sizes)
                quit()
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, chunk_sizes, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, chunk_sizes, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, chunk_sizes, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, n_gpu, verbose, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu0_bsz = gpu0_bsz
        self.verbose = verbose
        self.ngpu = n_gpu
        if gpu0_bsz==0:
            self.ngpu -= 1
        self.start_gpu_id = 1 # used for last batch

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        if self.gpu0_bsz == 0:
            device_ids = self.device_ids[1:]
        else:
            device_ids = self.device_ids
        inputs, kwargs = self.scatter(inputs, kwargs, device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        if 0 not in self.used_device_ids:
            replicas = self.replicate(self.module, self.device_ids[:1+len(self.used_device_ids)])
            replicas = replicas[1:]
        else:
            replicas = self.replicate(self.module, self.used_device_ids)
        if self.verbose:
            for ii in range(len(inputs)):
                print('device %d bz %d'%(self.used_device_ids[ii], inputs[ii][0].shape[0]))
        outputs = self.parallel_apply(replicas, self.used_device_ids, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def parallel_apply(self, replicas, device_ids, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, device_ids[:len(inputs)])

    def scatter(self, inputs, kwargs, device_ids):
        bsz = inputs[0].size(self.dim)
        num_dev = len(device_ids)
        avg_unit = bsz // num_dev
        if self.gpu0_bsz == 0: # in this case, devices_ids does not include gpu0
            if avg_unit == 0: # bsz<num_dev
                chunk_sizes = [1] * bsz
                self.used_device_ids = device_ids[:bsz]
                return scatter_kwargs(inputs, kwargs, device_ids[:bsz], chunk_sizes, dim=self.dim)
            else:
                chunk_sizes = [avg_unit] * num_dev
                delta = bsz - sum(chunk_sizes)
                for i in range(delta):
                    chunk_sizes[i] +=1
                self.used_device_ids = device_ids
                return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
        else: # in this case gpu_devices include gpu0
            if avg_unit == 0:
                chunk_sizes = [1] * bsz
                self.used_device_ids = device_ids[1:bsz+1]
                return scatter_kwargs(inputs, kwargs, device_ids[1:bsz+1], chunk_sizes, dim=self.dim)
            elif avg_unit < self.gpu0_bsz:
                chunk_sizes = [avg_unit] * num_dev
                delta = bsz - sum(chunk_sizes)
                for i in range(delta):
                    chunk_sizes[i+1] +=1
                self.used_device_ids = device_ids
                return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
            else:
                bsz_unit = (bsz - self.gpu0_bsz) // (num_dev - 1)
                chunk_sizes = [self.gpu0_bsz] + [bsz_unit] * (num_dev - 1)
                delta = bsz - sum(chunk_sizes)
                iid = self.start_gpu_id
                for i in range(delta):
                    iid +=  i  # exclude gpu0
                    if iid>=self.ngpu:
                        iid = 1
                    chunk_sizes[iid] += 1
                self.start_gpu_id = 1 if (iid+1)>=self.ngpu else iid+1
                self.used_device_ids = device_ids
                return scatter_kwargs(inputs, kwargs, device_ids, chunk_sizes, dim=self.dim)
