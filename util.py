# set up dataloader for conditional vae
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn.functional as F

class Maestro(Dataset):

    # def __init__(self, music_file, curve_file):
    def __init__(self, music_file, input_height):
        self.data = pd.read_csv(music_file, header=None)
        self.input_height = input_height
        # self.curve = pd.read_csv(curve_file,header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # c = torch.from_numpy(self.curve.iloc[idx].values)
        seq = torch.from_numpy(self.data.iloc[idx].values)
        doh = F.one_hot(seq, self.input_height).type(torch.FloatTensor)
        # coh = F.one_hot(c, cond_height).type(torch.FloatTensor)
        # return (doh, coh)
        return doh
