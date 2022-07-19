import math

import pandas
import argparse
import torch
from swda_gmm_dataset import SWDAGMMDataset
from icarus_util import *
from icarus_min_trainer import obtain_wav2vec
from icarus_gmm_trainer import get_wav2vec_rms, ce_da_loss, get_acoustic_rms

from icarus_mode_gmm import ModeGPTLSTMGMM
from icarus_heatmap import ModeGPTLSTMHeatmap
from icarus_insert_bc import FeedbackInserter
from icarus_mae import MAECalculator

import json

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

win_size = 100
f_shift = 50
df_p = pandas.DataFrame([])
df_outputs = pandas.DataFrame([])
df_rms = pandas.DataFrame([])
model_type = "Time"
gmm_mode = "mean"

include_preds = False

EXP_DIR = "experiments/"

all_st_ts = []
all_pred_ts = []
all_true_st_ts = []
all_w_vecs = []
all_rms = []

all_da = []
all_pred_da = []
D_MAX = 5
D_MAX_PRED = 1



def save_df(st_ts, out):
    global df_p
    new_row = {}
    # st_ts = st_ts.detach().cpu().numpy()
    for j in range(len(st_ts)):
        for i in range(len(st_ts[0])):
            new_row.update({"st_ts_" + str(i): st_ts[j, i].item()})
            if len(out.shape) == 3:
                new_row.update({"pred_st_ts_" + str(i): out[j, i, 0].item()})
                new_row.update({"pred_st_std_" + str(i): out[j, i, 1].item()})
            else:
                new_row.update({"pred_st_ts_" + str(i): out[j, i].item()})
        df_p = df_p.append(new_row, ignore_index=True)
        new_row = {}


def save_rms(rms):
    global df_rms
    new_row = {}
    for j in range(len(rms)):
        for i in range(len(rms[0])):
            new_row.update({"rms_" + str(i): rms[j, i].item()})
        df_rms = df_rms.append(new_row, ignore_index=True)
        new_row = {}


def forward_to_time_heatmap(out):
    global D_MAX
    preds = np.zeros((out.shape[0], out.shape[1]))
    T = out.shape[-1]
    i_s = np.array([(i / T) * 2 * D_MAX for i in range(T)])
    h_inds = np.argmax(out, axis=-1)
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            preds[i, j] = i_s[h_inds[i, j]]
    return preds


def forward_to_time(out):
    global D_MAX
    T = D_MAX * 16 * 2
    mu = out['mu'].detach().cpu().numpy()
    sigma = out['sigma'].detach().cpu().numpy()
    h = out['h'].detach().cpu().numpy()
    hs_sum = np.sum(h, axis=-1, keepdims=True)
    hs_sum = np.where(hs_sum > 0, hs_sum, 1e-10)
    h = np.divide(h, hs_sum)
    if gmm_mode == "hmax":
        mu_ind = np.argmax(h, axis=-1)
        preds = np.zeros((mu.shape[0], mu.shape[1]))
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                preds[i, j] = mu[i, j, mu_ind[i, j]]
        return preds, (mu, sigma, h)
    elif gmm_mode == "approx_mean":
        preds = np.zeros((mu.shape[0], mu.shape[1]))
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                preds[i][j] = np.sum(mu[i, j, :] * h[i, j, :])
        return preds, (mu, sigma, h)


    mu_new = np.repeat(mu[:, :, :, np.newaxis], T, axis=3)
    sigma_new = np.repeat(sigma[:, :, :, np.newaxis], T, axis=3)
    sigma_new = sigma_new + 1e-10
    h_new = np.repeat(h[:, :, :, np.newaxis], T, axis=3)

    def calc_gaussian(mu, sigma, i_s):
        global D_MAX
        # sigma = sigma + 1e-7
        gs = np.exp((-1 / (2 * np.square(sigma))) * np.square(mu - i_s)) * (1 / (sigma * math.sqrt(2 * math.pi)))
        gs_s = np.sum(gs, axis=-1, keepdims=True)
        gs_s = np.where(gs_s > 0, gs_s, 1e-10)
        gs = np.divide(gs, gs_s)
        return gs

    i_s = np.array([(i / T) * 2 * D_MAX for i in range(T)])
    gaussians = calc_gaussian(mu_new, sigma_new, i_s)
    preds = h_new * gaussians
    if gmm_mode == "mean":
        # preds = preds + 1e-20
        # preds = preds / np.sum(preds, axis=-1, keepdims=True)
        preds = np.sum(preds * i_s, axis=-1)
        preds = np.sum(preds, axis=-1)
        # preds = np.sum(preds, axis=-1) / mu.shape[-1]
        preds = np.where(preds < 0.05, 0, preds)
    else:
        preds = np.sum(preds, axis=-1)
        preds = np.argmax(preds, axis=-1)
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                preds[i, j] = i_s[preds[i, j]]
    # print(preds.shape)
    return preds, (mu, sigma, h)

def save_outputs_heatmap(out, st_ts):
    global df_outputs
    for i in range(st_ts.shape[0]):
        for j in range(st_ts.shape[1]):
            new_row = {"st_ts": st_ts[i, j]}
            h_t = out[i, j, :]
            for k, h in enumerate(h_t):
                new_row.update({"h_" + str(k): h})
            df_outputs = df_outputs.append(new_row, ignore_index=True)

def save_outputs(all_outs, st_ts):
    global df_outputs
    mu, sigma, h = all_outs
    for i in range(st_ts.shape[0]):
        for j in range(st_ts.shape[1]):
            new_row = {"st_ts": st_ts[i, j]}
            h_t = h[i, j, :]
            inds = (np.where(h_t != 0)[0]).tolist()
            for k, idx in enumerate(inds):
                new_row.update({"index_" + str(k): idx})
                new_row.update({"mu_" + str(k): mu[i, j, idx]})
                new_row.update({"sigma_" + str(k): sigma[i, j, idx]})
                new_row.update({"h_" + str(k): h_t[idx]})
            df_outputs = df_outputs.append(new_row, ignore_index=True)


def predict_and_save(model, audio_ft, st_ts, trans, w_inds, w_vec, true_st_ts, da):
    global df_p, all_st_ts, all_pred_ts, all_w_vecs, all_da, all_pred_da
    if model.mode == 3:
        rmss = audio_ft[:, :, -1].squeeze(0)
        save_rms(rmss)
    audio_ft = audio_ft.to(device)
    w_inds = w_inds.to(device)
    out = model(audio_ft, model.init_hidden(audio_ft.shape[0]), trans, w_inds)
    if isinstance(model, ModeGPTLSTMGMM):
        pred_st_ts, all_outs = forward_to_time(out)
        if include_preds:
            save_outputs(all_outs, st_ts)
    elif isinstance(model, ModeGPTLSTMHeatmap):
        out = out.detach().cpu().numpy()
        pred_st_ts = forward_to_time_heatmap(out)
        if include_preds:
            save_outputs_heatmap(out, st_ts)
    else:
        pred_st_ts = out[:, :, 0].detach().cpu().numpy()

    st_ts = st_ts.detach().cpu().numpy()
    all_st_ts.append(st_ts)
    all_pred_ts.append(pred_st_ts)
    all_w_vecs.append(w_vec)
    all_true_st_ts.append(true_st_ts)
    all_da.append(da)

    if "d_act" in out:
        all_pred_da.append(ce_da_loss(out["d_act"], da.to(device), w_vec.to(device)).detach().cpu().item())
    # if include_preds:
    #     save_df(st_ts, pred_st_ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--exp_num", type=int)
    parser.add_argument("--fid", type=str, default="test_20")
    parser.add_argument("--convo", type=str, default="None")
    parser.add_argument("--pred_csv", type=str, default="predictions.csv")
    parser.add_argument("--add_pred", action="store_true")
    parser.add_argument("--add_wav", action="store_true")
    parser.add_argument("--add_zero_diff", action="store_true")
    parser.add_argument("--dmax", type=int, default=2)
    parser.add_argument("--dmax_pred", type=float, default=1)
    parser.add_argument("--gmm_mode", type=str, default="mean")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--bottom", action="store_true")
    args = parser.parse_args()

    EXP_DIR = EXP_DIR + str(args.exp_num) + "/"

    include_preds = args.add_pred
    include_wav = args.add_wav

    D_MAX = args.dmax
    D_MAX_PRED = args.dmax_pred
    gmm_mode = args.gmm_mode

    model_path = args.model_path if args.model_path is not None else EXP_DIR + "model-after-train.pt"
    model = torch.load(model_path, map_location=torch.device(device))
    model.eval()
    model.to(device)

    """
        Determine model type
    """
    if isinstance(model, ModeGPTLSTMGMM):
        model_type = "GMM"
    elif isinstance(model, ModeGPTLSTMHeatmap):
        model_type = "Heatmap"
    print(model_type, model.mode)

    fid = args.fid
    conv = args.convo
    if include_preds:
        plot_dataset = SWDAGMMDataset([], f_id=fid, conv=conv, debug=True, d_max=D_MAX, size_limit=10, st_size_limit=5, mode=0)
    elif fid == "train_200":
        plot_dataset = SWDAGMMDataset([], f_id=fid, conv=conv, debug=True, d_max=D_MAX, size_limit=3)
    else:
        plot_dataset = SWDAGMMDataset([], f_id=fid, conv=conv, debug=True, d_max=D_MAX, mode=0)

    fb = FeedbackInserter(EXP_DIR)
    # mc = MetricsCalculator(D_MAX, D_MAX_PRED, zeros=args.zero, bottom=args.bottom)
    # zmc = ZeroDiffCalculator(D_MAX_PRED, args.sigma)
    mc_ttee = MAECalculator(D_MAX)

    for i in range(len(plot_dataset.wavs)):
        if include_wav:
            fb.add_to_wav(plot_dataset.wavs[i])
        if model.mode == 3:
            plot_dataset.wavs[i] = get_wav2vec_rms(plot_dataset.wavs[i])
        elif model.mode == 4:
            plot_dataset.wavs[i] = get_acoustic_rms(plot_dataset.wavs[i])
        else:
            plot_dataset.wavs[i] = obtain_wav2vec(plot_dataset.wavs[i])

    kwargs = {'num_workers': 0, 'pin_memory': False} if device == 'cuda:0' else {}  # needed for using datasets on gpu
    plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=20, shuffle=False,
                                              **kwargs)

    pred_start = []
    real_start = []
    for (d, labels) in plot_loader:
        d = d.to(device)
        if include_wav:
            fb.edit_wavs(labels['w_inds'], labels['chanl'], labels['beg'], labels['end'])
        predict_and_save(model, d, labels['start_ts'], labels['transcript'], labels['word_inds'], labels['gmm_w'],
                         labels['true_start_ts'], labels['dialog_act'])

    all_st_ts = np.vstack(all_st_ts)
    all_pred_ts = np.vstack(all_pred_ts)
    all_w_vecs = np.vstack(all_w_vecs)
    all_true_st_ts = np.vstack(all_true_st_ts)
    all_da = np.vstack(all_da)
    if include_preds:
        # df_p.to_csv(EXP_DIR + args.pred_csv, index=False)
        df_rms.to_csv(EXP_DIR + "rms_vals.csv", index=False)
        # if model_type == "GMM":
        #     df_outputs.to_csv(EXP_DIR + args.pred_csv.replace(".csv", "_GMM_outputs.csv"), index=False)
        # elif model_type == "Heatmap":
        #     df_outputs.to_csv(EXP_DIR + args.pred_csv.replace(".csv", "_Heatmap_outputs.csv"), index=False)
        if include_wav:
            fb.add_bc(all_pred_ts)
    else:
        res = 16
        threshs = [x / res for x in range(0, D_MAX * res + 1, 1)]
        mets = mc_ttee.calc_ttees(all_true_st_ts, all_pred_ts, all_w_vecs, threshs)
        json.dump(mets, open(EXP_DIR + args.pred_csv.replace(".csv", "_ttee.json"), "w+"))
        # all_da_0 = np.where(all_da == 0, 1, 0)
        # all_da_1 = np.where(all_da == 1, 1, 0)
        # mets = mc_ttee.calc_ttees(all_true_st_ts, all_pred_ts, all_da_0 * all_w_vecs, threshs)
        # json.dump(mets, open(EXP_DIR + args.pred_csv.replace(".csv", "_da0_ttee.json"), "w+"))
        # mets = mc_ttee.calc_ttees(all_true_st_ts, all_pred_ts, all_da_1 * all_w_vecs, threshs)
        # json.dump(mets, open(EXP_DIR + args.pred_csv.replace(".csv", "_da1_ttee.json"), "w+"))

        if len(all_pred_da):
            final_ce = sum(all_pred_da) / len(all_pred_da)
            json.dump({"da_loss": final_ce}, open(EXP_DIR + args.pred_csv.replace(".csv", "_dact.json"), "w+"))
        # if args.sweep and not args.add_zero_diff:
        #     threshs = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
        #                    0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 1.25,
        #                    1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2]
        #     mets = mc.sweep_thresholds(threshs, all_st_ts, all_pred_ts, all_w_vecs, all_true_st_ts)
        #     for i in range(len(threshs)):
        #         if args.bottom:
        #             json.dump(mets[i], open(EXP_DIR + "bottom-" + args.pred_csv.replace(".csv", "_" + str(threshs[i])
        #                                                                                 + "_metrics.json"), "w+"))
        #         else:
        #             json.dump(mets[i], open(EXP_DIR + args.pred_csv.replace(".csv", "_" + str(threshs[i])
        #                                                                                 + "_metrics.json"), "w+"))
        # else:
        #     if args.add_zero_diff:
        #         metrics = {}
        #         if args.sweep:
        #             zero_threshs = [0, 0.001, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        #             sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        #             for z in zero_threshs:
        #                 for s in sigmas:
        #                     zmc.set_zero(z)
        #                     zmc.set_sigma(s)
        #                     acc = zmc.calc_zero_class_acc(all_pred_ts, all_st_ts)
        #                     metrics.update({"acc-" + str(z) + "-" + str(s): acc})
        #         else:
        #             acc = zmc.calc_zero_class_acc(all_pred_ts, all_st_ts)
        #             metrics.update({"acc": acc})
        #             print("Classification Accuracy ===", acc)
        #         json.dump(metrics, open(EXP_DIR + args.pred_csv.replace(".csv", "_" + "_zero_acc.json"), "w+"))
        #     else:
        #         met = mc.calculate_metrics(all_st_ts, all_pred_ts, all_w_vecs, all_true_st_ts)
        #         json.dump(met, open(EXP_DIR + args.pred_csv.replace(".csv", "_" + str(D_MAX_PRED) + "_metrics.json"), "w+"))
