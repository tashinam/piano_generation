import IPython.display
import pretty_midi
import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

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


def display_midi(mel):
    pno = torch.tensor([0, 0.5, 0, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0, 0.5]).repeat(len(mel), 8).T + 1
    indices = torch.tensor([mel]) - 21
    scale = torch.zeros((88, len(mel)))
    scale.scatter_(0, indices, 1)

    plt.figure(figsize=(len(mel) / 2.5, (torch.max(indices) - torch.min(indices)) / 6))
    plt.imshow(pno, origin='lower', aspect='auto', vmin=0.5, cmap='viridis_r')
    plt.imshow(scale, origin='lower', aspect='auto', alpha=scale, cmap='spring')
    plt.ylim(torch.min(indices) - 0.5, torch.max(indices) + 0.5)
    plt.xticks([])
    plt.yticks([])
    plt.vlines(np.array(range(0, len(mel), 8)) - 0.5, ymin=0, ymax=88, linewidth=0.5)

    return plt.show()

def play_midi(mel):
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    inst = pretty_midi.Instrument(program=0)
    pm.instruments.append(inst)
    velocity = 80
    for pitch, start, end in zip(mel, np.arange(0,len(mel),0.3), np.arange(0,len(mel),0.3)+0.5):
        inst.notes.append(pretty_midi.Note(velocity, pitch, start, end))
    return IPython.display.Audio(pm.fluidsynth(fs=44100), rate=44100)