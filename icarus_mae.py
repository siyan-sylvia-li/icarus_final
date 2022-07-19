import collections
import glob
import math
import pickle
import random

import pandas
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from icarus_util import *
from collections import Counter

import json
from sklearn.metrics import confusion_matrix, classification_report


D_MAX = 5


def calc_mean_ci(arr):
    arr_np = np.array(arr)
    mean = np.mean(arr_np)
    ci = 1.96 * np.std(arr_np) / (math.sqrt(len(arr_np)))
    return mean, ci

class MAECalculator:
    def __init__(self, dmax, resolution=16):
        self.dmax = dmax
        self.res = resolution
        self.resolution = 1 / resolution
        self.abs_diff = None

    def in_range(self, pred, t):
        return pred == t

    def gen_gold_thresh(self):
        threshs = [-x / self.res for x in range(0, self.dmax * self.res, 1)]
        threshs.extend([x / self.res for x in range(0, self.res + 1, 1)])
        threshs = list(set(threshs))
        threshs.sort()
        threshs.insert(0, -2)
        return threshs

    def convert_bucket(self, x):
        if x == 0:
            return 0
        up = self.resolution * int(math.ceil(x / self.resolution))
        down = self.resolution * int(x / self.resolution)
        if abs(x - up) > abs(x - down):
            return down
        elif abs(x - up) == abs(x - down):
            return random.choice([up, down])
        return up

    def make_buckets(self, ts: np.array):
        ts_bucket = np.zeros(ts.shape)
        for i in range(ts.shape[0]):
            for j in range(ts.shape[1]):
                ts_bucket[i][j] = self.convert_bucket(ts[i][j])
        return ts_bucket

    def mae_true(self, all_st_ts, all_pred_ts, all_w_vecs):
        gold_threshs = self.gen_gold_thresh()
        print(gold_threshs)
        avg_preds = [[] for _ in range(len(gold_threshs))]
        thresh_sum = [[] for _ in range(len(gold_threshs))]

        zero_counter = np.zeros(all_st_ts.shape)
        for i in range(zero_counter.shape[0]):
            for j in range(1, zero_counter.shape[1]):
                if all_st_ts[i][j - 1] > 0 and all_st_ts[i][j] == 0:
                    zero_counter[i][j: j + 20] = j

        for i in range(all_st_ts.shape[0]):
            for j in range(all_st_ts.shape[1]):
                if all_w_vecs[i][j]:
                    st = all_st_ts[i][j]
                    pd = all_pred_ts[i][j]
                    if st == 0 and zero_counter[i][j]:
                        zero_starter = j - zero_counter[i][j]
                        ind = gold_threshs.index(self.convert_bucket(zero_starter * 0.05))
                        thresh_sum[ind].append(abs(st - pd))
                        avg_preds[ind].append(pd)
                    elif st > 0 and (-st) in gold_threshs:
                        # This is the before times
                        ind = gold_threshs.index(-st)
                        thresh_sum[ind].append(abs(st - pd))
                        avg_preds[ind].append(pd)
        gold_dict = {}
        for i, t in enumerate(gold_threshs):
            gt_m, gt_c = calc_mean_ci(thresh_sum[i])
            gold_dict.update({"gold|" + str(t): gt_m})
            gold_dict.update({"goconf|" + str(t): gt_c})
            ga_m, ga_c = calc_mean_ci(avg_preds[i])
            gold_dict.update({"gavg|" + str(t): ga_m})
            gold_dict.update({"gaconf|" + str(t): ga_c})
            gold_dict.update({"gsupport|" + str(t): len(thresh_sum[i])})

        return gold_dict

        # true_occur = np.where(all_st_ts == thresh, np.ones(all_st_ts.shape), np.zeros(all_st_ts.shape))
        # # true_occur = np.where(all_w_vecs > 0, true_occur, np.zeros(all_st_ts.shape))
        # true_occur = np.where(all_pred_ts < self.dmax, true_occur, 0)
        # curr_abs_diff = np.where(true_occur, np.abs(all_pred_ts - all_st_ts), np.zeros(all_st_ts.shape))
        # support = np.sum(true_occur * all_w_vecs)
        # avg = np.sum(np.where(true_occur, all_pred_ts, 0) * all_w_vecs) / support
        # ttee = np.sum(curr_abs_diff * all_w_vecs) / support
        #### ONE BY ONE APPROACH
        # support = 0
        # ttee = 0
        # avgs = []
        # for i in range(all_st_ts.shape[0]):
        #     for j in range(all_st_ts.shape[1]):
        #         if all_st_ts[i][j] == thresh and all_w_vecs[i][j] > 0:
        #             ttee = ttee + min(self.dmax, abs(all_pred_ts[i][j] - all_st_ts[i][j])) * all_w_vecs[i][j]
        #             support += all_w_vecs[i][j]
        #             avgs.append(all_pred_ts[i][j])
        # if support == 0:
        #     return float("inf"), support, avgs
        # ttee = ttee / support

    def mae_pred(self, st_ts, pred_ts, all_w_vecs, thresh):
        # true_occur = np.where(all_pred_ts == thresh, 1, 0)
        # true_occur = np.where(all_w_vecs, true_occur, 0)
        # abs_diff = np.abs(all_st_ts - all_pred_ts)
        # abs_diff = np.where(true_occur, np.minimum(abs_diff, self.dmax), np.zeros(all_pred_ts.shape))
        # for i in range(all_pred_ts.shape[0]):
        #     for j in range(all_pred_ts.shape[1]):
        #         if true_occur[i][j] and 0 < all_pred_ts[i][j] < 0.5:
        #             print((all_st_ts[i][j], all_pred_ts[i][j], abs_diff[i][j]))
        # support = np.sum(true_occur)
        # ttee = np.sum(abs_diff) / support

        true_occur = np.where(pred_ts == thresh, 1, 0)
        true_occur = np.where(all_w_vecs > 0, true_occur, 0)
        # true_occur = np.where(st_ts < self.dmax, true_occur, 0)
        # curr_abs_diff = np.where(true_occur, np.abs(pred_ts - st_ts), np.zeros(st_ts.shape))
        ttee_pred = []
        ttee_avg = []
        for i in range(st_ts.shape[0]):
            for j in range(st_ts.shape[1]):
                if pred_ts[i][j] == thresh and all_w_vecs[i][j]:
                    ttee_pred.append(abs(st_ts[i][j] - thresh))
                    ttee_avg.append(st_ts[i][j])
        support = len(ttee_avg)
        assert len(ttee_pred) == len(ttee_avg)
        print("=" * 40)
        print(thresh)
        print(ttee_pred[:30], calc_mean_ci(ttee_pred))
        print(ttee_avg[:30], calc_mean_ci(ttee_avg))
        print("=" * 40)
        pm, pc = calc_mean_ci(ttee_pred)
        pam, pac = calc_mean_ci(ttee_avg)
        metrics = {}

        metrics.update({"pred-" + str(thresh): float(pm)})
        metrics.update({"pci-" + str(thresh): float(pc)})
        metrics.update({"psupport-" + str(thresh): int(support)})
        metrics.update({"pavg-" + str(thresh): float(pam)})
        metrics.update({"paci-" + str(thresh): float(pac)})

        #### ONE BY ONE

        # support = 0
        # ttee = 0
        # for i in range(all_pred_ts.shape[0]):
        #     for j in range(all_pred_ts.shape[1]):
        #         if all_pred_ts[i][j] == thresh and all_w_vecs[i][j] > 0:
        #             ttee = ttee + min(self.dmax, abs(all_pred_ts[i][j] - all_st_ts[i][j])) * all_w_vecs[i][j]
        #             support += all_w_vecs[i][j]
        # if support == 0:
        #     return float("inf"), support
        # ttee = ttee / support
        return metrics

    def calc_ttees(self, all_st_ts, all_pred_ts, all_w_vecs, threshs):
        assert all_st_ts.shape == all_pred_ts.shape
        metrics = {}
        # Flatten out the top
        all_st_ts = np.minimum(all_st_ts, self.dmax)
        all_pred_ts = np.minimum(all_pred_ts, self.dmax)
        pred_ts = self.make_buckets(all_pred_ts)
        st_ts = self.make_buckets(all_st_ts)
        self.abs_diff = np.where(st_ts <= self.dmax, np.abs(st_ts - pred_ts), np.zeros(all_st_ts.shape))
        golds = self.mae_true(st_ts, pred_ts, all_w_vecs)
        metrics.update(golds)
        for t in threshs:
            # t = self.convert_bucket(t)
            ttee_pred_dict = self.mae_pred(st_ts, pred_ts, all_w_vecs, t)
            metrics.update(ttee_pred_dict)
        return metrics
