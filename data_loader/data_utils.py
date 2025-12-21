import numpy as np
import torch
from torch.utils.data import Dataset
from data_loader import aug


class dataset(Dataset):
    def __init__(self, list_data, transform=None):
        self.seq_data = list_data['data'].tolist()
        self.labels   = list_data['labels'].tolist()
        self.transforms = transform or aug.Compose([aug.Retype()])

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        y   = float(self.labels[item])
        seq = self.transforms(seq)
        return seq, torch.tensor(y, dtype=torch.float32)

