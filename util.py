# set up dataloader for conditional vae
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np


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


def note2scale(x):
    n2s = {'0': 60, '1': 62, '2': 64, '3': 65, '4': 67, '5': 69, '6': 71,
           '7': 72, '8': 74, '9': 76, '10': 77, '11': 79, '12': 81, '13': 83,
           '14': 84}
    return n2s.get(str(int(x)), x)


def scale2note(x):
    s2n = {'60': 0, '62': 1, '64': 2, '65': 3, '67': 4, '69': 5, '71': 6,
           '72': 7, '74': 8, '76': 9, '77': 10, '79': 11, '81': 12, '83': 13,
           '84': 14}
    return s2n.get(str(int(x)), x)


def best_fit_slope_and_intercept(xs, ys):
    m = (((np.mean(xs) * np.mean(ys)) - np.mean(xs * ys)) / ((np.mean(xs) * np.mean(xs)) - np.mean(xs * xs)))
    b = np.mean(ys) - m * np.mean(xs)
    return m, b
