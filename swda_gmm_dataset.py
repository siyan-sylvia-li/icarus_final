import random

from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import h5py

"""
    Global list of parameters
"""
params = {"w_size": 100, "f_shift": 50, "utt_len": 1200}
acts_dict = [x[:-1] for x in open("dialogue_acts.txt", "r").readlines()]
randomize = True
f_id = ""

random.seed(42)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

wrong_files = ['sw2247', 'sw2262', 'sw2290', 'sw2485', 'sw2521', 'sw2533', 'sw2543', 'sw2617', 'sw2627', 'sw2631',
               'sw2684', 'sw2844', 'sw2930', 'sw2954', 'sw2968', 'sw3039', 'sw3050', 'sw3129', 'sw3144', 'sw2073',
               'sw2616']


def calc_gaussian(st_ts, T, d_max):
    gs = np.zeros((len(st_ts), T))
    sigma = d_max / 4
    i_s = np.array([(i / T) * 2 * d_max for i in range(T)])
    for i in range(gs.shape[0]):
        gs[i, :] = np.exp((-1 / (2 * sigma * sigma)) * np.square(i_s - st_ts[i]))
    return gs


def calc_occur_weight(x, d_max, mode):
    x1 = np.where(x < d_max, x, np.zeros(x.shape))
    x1 = np.where(x1 > 0, np.ones(x.shape), np.zeros(x.shape))
    x_dmax = np.copy(x)

    # If we have a frame shift of 50 ms
    shift = 2 * d_max * 20
    # Iterate through all the time steps for each example
    for i in range(x.shape[0]):
        for j in range(x.shape[1] - 1):
            # If this index is non-zero but everything is zero after, as in the initiation
            # Then this index is our base, and we say everything D_MAX before it is 1
            if x[i, j] > 0 and x[i, j + 1] == 0:
                x1[i, max(0, j - shift): j] = 1
                for k in range(j - 1, max(j - shift // 2 - 1, 0), -1):
                    if x_dmax[i, k] == 0:
                        x_dmax[i, k] = x_dmax[i, k + 1] + 0.05
                if mode == 2:
                    x1[i, j: min(j + 20, x.shape[1])] = 0.5
    return x1, x_dmax


class SWDAGMMDataset(Dataset):
    # Wrapper for updated SWDA Dataset
    def __init__(self, conv_list, f_id, size_limit=-1, st_size_limit=-1,
                 conv='None', debug=False, d_max=5, mode=0, process=None, resolution=4):
        self.conv_list = conv_list
        self.conv_len_dict = OrderedDict()
        self.size_limit = size_limit
        self.f_id = f_id
        self.st_ts = []
        self.true_st_ts = []
        # self.bucket_st_ts = []
        # self.bucket_end_ts = []
        # self.boundaries = torch.tensor([0, 0.5, 1, 2, 4, float("inf")])
        # self.end_boundaries = torch.tensor([0, 3, 10, 20, float("inf")])
        self.end_ts = []
        self.d_act = []
        self.wavs = []
        # Which wav file are we talking about
        self.w_inds = []
        self.indices = []
        self.w_vecs = []
        self.transcripts = []
        self.word_inds = []
        self.st_size_limit = st_size_limit
        self.conv = conv
        self.debug = debug
        self.d_max = d_max
        self.tanh = torch.nn.Tanh()
        self.gmm_weights = []
        self.process = process
        self.resolution = resolution
        self.mode = mode  # 0 for original, 1 for 2Dmax only, 2 for 2Dmax + 1 sec, 3 for 2Dmax with different adjustments
        # According to equation in paper
        self.T = self.d_max * 2 * self.resolution
        self.g = []
        self.load_conv_len(f_id, conv)

    def load_conv_len(self, f_id, conv):
        try:
            f = h5py.File("/scr/biggest/siyanli/icarus/full_data_wav2vec/" + f_id + ".hdf5", "r")
        except FileNotFoundError:
            f = h5py.File("full_data_wav2vec/" + f_id + ".hdf5", "r")
        if conv == 'None':
            all_files = f.keys()
        else:
            all_files = [conv]
        if self.size_limit:
            all_files = list(all_files)[:self.size_limit]
        # all_files = list(all_files)[3:]
        print("Will be loading: ", len(all_files), "files.")
        self.conv_list = all_files

        for f_st in all_files:
            self.load_data(f, f_st)

    def load_data(self, f, f_st):
        # print("loading... ", f_st)
        dialogue_grp = f[f_st]
        new_st_ts = dialogue_grp['st_ts'][:]
        orig_st_ts = dialogue_grp['st_ts'][:]
        # Capping + random sampling between D_MAX and 2D_MAX
        new_st_ts = np.where(new_st_ts > self.d_max, self.d_max, new_st_ts)
        gmm_weights, new_st_ts_dmax = calc_occur_weight(new_st_ts, self.d_max, self.mode)
        if self.mode != 0:
            new_st_ts_dmax = np.where(new_st_ts_dmax > 0, new_st_ts_dmax, self.d_max)
            new_st_ts = np.where(gmm_weights == 1, new_st_ts_dmax, new_st_ts)
        else:
            new_st_ts = np.minimum(new_st_ts, self.d_max)
        gmm_weights = np.where(gmm_weights == 0.5, 1, gmm_weights)

        new_d_act = dialogue_grp['d_act'][:]
        new_d_act = np.where(new_d_act != acts_dict.index("b"), 1, 0)

        new_st_ts = new_st_ts.tolist()
        orig_st_ts = orig_st_ts.tolist()
        gmm_weights = gmm_weights.tolist()

        new_end_ts = dialogue_grp['end_ts'][:].tolist()
        new_diff = dialogue_grp['end_ts'][:] - dialogue_grp['st_ts'][:]
        for i in range(new_diff.shape[0]):
            for j in range(1, new_diff.shape[1]):
                if new_diff[i][j] == 0 and new_diff[i][j - 1] > 0:
                    new_diff[i][j] = new_diff[i][j - 1]
        new_diff = np.where(new_diff < 1, 1, 0)
        new_d_act = np.where(new_diff, new_d_act, 1)
        # new_audio_ft = dialogue_grp['audio_ft'][:].tolist()
        new_indices = dialogue_grp['indices'][:].tolist()
        new_word_inds = dialogue_grp['words'][:].tolist()
        w_ind = len(self.wavs)
        new_wav = torch.FloatTensor(dialogue_grp['wavs'][:])
        if f_st in wrong_files:
            new_wav = torch.cat([new_wav[1].unsqueeze(0), new_wav[0].unsqueeze(0)], dim=0)
        self.wavs.append(new_wav)

        new_trans = dialogue_grp['transcript'][:].tolist()
        new_trans = [[m.decode("utf-8") for m in new_trans[0] if len(m)],
                     [m.decode("utf-8") for m in new_trans[1] if len(m)]]
        self.transcripts.append(new_trans)

        new_d_act = new_d_act.tolist()
        lim = self.st_size_limit if self.st_size_limit > 0 else len(new_st_ts)

        for i in range(lim):
            self.w_inds.append(w_ind)
            self.true_st_ts.append(torch.FloatTensor(orig_st_ts[i]))
            if self.process == "tanh":
                self.st_ts.append(self.tanh(torch.FloatTensor(new_st_ts[i])))
            else:
                self.st_ts.append(torch.FloatTensor(new_st_ts[i]))
            self.g.append(calc_gaussian(new_st_ts[i], self.T, self.d_max))
            # self.bucket_st_ts.append(torch.bucketize(self.st_ts[-1], self.boundaries))
            self.end_ts.append(torch.FloatTensor(new_end_ts[i]))
            # self.bucket_end_ts.append(torch.bucketize(self.end_ts[-1] - self.st_ts[-1], self.end_boundaries))
            self.indices.append(new_indices[i])
            self.word_inds.append(torch.IntTensor(new_word_inds[i]))
            self.d_act.append(torch.LongTensor(new_d_act[i]))
            self.gmm_weights.append(torch.FloatTensor(gmm_weights[i]))

    def wav_len(self):
        return len(self.wavs)

    def check_rms(self):
        anomaly_total = 0
        anomaly_inds = []
        wav_flipped = [False] * len(self.wavs)
        wav_ans = [(0, 0)] * len(self.wavs)
        print(self.st_ts[0], self.st_ts[1], torch.equal(self.st_ts[0], self.st_ts[1]))
        for idx in tqdm(range(len(self.st_ts))):
            start_ts = self.true_st_ts[idx]
            chanl, beg, end = self.indices[idx][0], self.indices[idx][1], self.indices[idx][2]
            other_chan = 0 if chanl == 1 else 1
            wav = self.wavs[self.w_inds[idx]]
            rmss = wav[:, beg: end, -1].squeeze()
            assert len(rmss.shape) == 2
            for i in range(rmss.shape[-1]):
                if start_ts[i] == 0:
                    wav_ans[self.w_inds[idx]] = (wav_ans[self.w_inds[idx]][0], wav_ans[self.w_inds[idx]][1] + 1)
                    if rmss[other_chan, i] < rmss[chanl, i]:
                        wav_ans[self.w_inds[idx]] = (wav_ans[self.w_inds[idx]][0] + 1, wav_ans[self.w_inds[idx]][1])

        # print(wav_ans[0])

        for i in range(len(self.wavs)):
            anoms = wav_ans[i]
            anom_ratio = anoms[0] / anoms[1]
            print(anom_ratio)
            if anom_ratio > 0.5:
                self.wavs[i] = torch.cat([self.wavs[i][1].unsqueeze(0), self.wavs[i][0].unsqueeze(0)], dim=0)
                anomaly_total += 1
                anomaly_inds.append((i, anom_ratio))
        print("Anomaly Percentage:", anomaly_total / len(wav_flipped))
        print([(self.conv_list[i[0]], i[1]) for i in anomaly_inds])

    def __len__(self):
        return len(self.st_ts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        """
            Convert raw audio file to embedding from wav2vec
        """
        chanl, beg, end = self.indices[idx][0], self.indices[idx][1], self.indices[idx][2]
        # Calculate the wavform of interest from 50 ms frame shifts
        # number of seconds times 16000 => samples
        # beg = int(beg * (50/1000) * 16000)
        # end = int(end * (50/1000) * 16000)
        wav = self.wavs[self.w_inds[idx]]
        chunk = wav[chanl, beg: end].squeeze()
        """
            Get list of words that occurred before this
        """
        # print(idx, self.word_inds[idx][0].item(), self.word_inds[idx][-1].item())
        transcript_chunk = " ".join(self.transcripts[self.w_inds[idx]][chanl]
                                    [max(int(self.word_inds[idx][0].item()), 0): int(
            self.word_inds[idx][-1].item()) + 1])
        if self.word_inds[idx][0].item() == -1:
            # Then we insert start of text token
            transcript_chunk = "<|endoftext|> " + transcript_chunk

        ret_word_inds = self.word_inds[idx] - self.word_inds[idx][0]
        if len(transcript_chunk.split(" ")) < ret_word_inds[-1]:
            print("ALERT!")
            print(len(transcript_chunk.split(" ")), ret_word_inds[-1])
            input()

        sample_l = {'start_ts': self.st_ts[idx],
                    'true_start_ts': self.true_st_ts[idx],
                    # 'end_ts': self.end_ts[idx],
                    'dialog_act': self.d_act[idx],
                    'transcript': transcript_chunk,
                    'word_inds': ret_word_inds,
                    'gmm_w': self.gmm_weights[idx],
                    'g': self.g[idx],
                    'w_inds': self.w_inds[idx],
                    'chanl': chanl,
                    'beg': beg,
                    'end': end}
        # 'bucket_start_ts': self.bucket_st_ts[idx],
        # 'bucket_end_ts': self.bucket_end_ts[idx]}
        return chunk, sample_l

    def get_feats_size(self):
        return self.wavs[0].shape[-1]
        # return 512
