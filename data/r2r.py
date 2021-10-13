from torch.utils.data import Dataset
import math
class R2RDataset(Dataset):
    def __init__(self, length, split, n_gpu, val_batch_size):

        if split=='train':
            self.len = length
            self.extra_num = 0
            self.rep_num = 0
            self.data = [i for i in range(length)]
        else: # specify repeated data
            self.len = math.ceil(1.0*length/(n_gpu*val_batch_size))*n_gpu*val_batch_size
            self.extra_num = self.len - length
            self.data = [i for i in range(length)] + [length-1]*self.extra_num

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def get_extra_data_num(self):
        return self.extra_num