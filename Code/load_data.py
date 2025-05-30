import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

def load_or_generate_data(file_path):
    if os.path.exists(file_path):
        print(f"Loading dataset from {file_path}")
        data = np.load(file_path)['data']
    else:
        print(f"Please generating new dataset and saving to {file_path}")
    return data


class NavierStokesDataset(Dataset):
    def __init__(self, data, input_steps=1, pred_steps=1):
        self.data = data
        self.input_steps = input_steps
        self.pred_steps = pred_steps

    def __len__(self):
        return len(self.data) * (self.data.shape[1] - self.input_steps - self.pred_steps)

    def __getitem__(self, idx):
        sample_idx = idx // (self.data.shape[1] - self.input_steps - self.pred_steps)
        time_idx = idx % (self.data.shape[1] - self.input_steps - self.pred_steps)
        x = self.data[sample_idx, time_idx:time_idx+self.input_steps]
        y = self.data[sample_idx, time_idx+self.input_steps:time_idx+self.input_steps+self.pred_steps]
        x = torch.FloatTensor(x).permute(0, 3, 1, 2)
        y = torch.FloatTensor(y).permute(0, 3, 1, 2)
        return x, y
