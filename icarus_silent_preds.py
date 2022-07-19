import pandas
import argparse
import torch
from swda_gmm_dataset import SWDAGMMDataset
from icarus_util import *
from icarus_insert_bc import FeedbackInserter
from icarus_mae import MAECalculator

import json
import librosa

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

win_size = 100
f_shift = 50
df_p = pandas.DataFrame([])
df_outputs = pandas.DataFrame([])
model_type = "Time"
gmm_mode = "mean"

include_preds = False

EXP_DIR = "experiments/"

all_st_ts = []
all_pred_ts = []
all_true_st_ts = []
all_w_vecs = []
D_MAX = 2
D_MAX_DIST = 14  # 700 milliseconds, because the frame shift is 50 ms;
# can change into other numbers for different thresholds
D_MAX_PRED = 0.001
RMS_THRESH = 1e-6

params = {"w_size": 100, "f_shift": 50, "utt_len": 1200}


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
            # new_row.update({"energy_" + str(i): audio_ft[j, i, -1].item()})
            # new_row.update({"d_act_" + str(i): d_act[j, i]})
        df_p = df_p.append(new_row, ignore_index=True)
        new_row = {}


def predict_and_save(audio_ft, st_ts, w_vecs, true_st_ts):
    """
    Predicts 0 for all "silence" (when RMS values are lower than the defined RMS_THRESH value,
    And predicts D_MAX for all non-silence.
    """
    global df_p, all_st_ts, all_pred_ts, all_w_vecs, all_true_st_ts, D_MAX, D_MAX_DIST
    all_st_ts.append(st_ts)
    all_w_vecs.append(w_vecs)
    all_true_st_ts.append(true_st_ts)
    # pred_st_ts = np.where(audio_ft > RMS_THRESH, 0, D_MAX)
    pred_st_ts = np.zeros(audio_ft.shape)
    assert pred_st_ts.shape == st_ts.shape
    for i in range(pred_st_ts.shape[0]):
        for j in range(pred_st_ts.shape[1]):
            if audio_ft[i][j] < RMS_THRESH:
                pred_st_ts[i][j] = 0
            else:
                pred_st_ts[i][j] = D_MAX
    real_preds = np.copy(pred_st_ts)
    for i in range(pred_st_ts.shape[0]):
        for j in range(pred_st_ts.shape[1]):
            if j < pred_st_ts.shape[1] - 1:
                if pred_st_ts[i][j] > 0 and pred_st_ts[i][j + 1] == 0:
                    real_preds[i][j: j + D_MAX_DIST] = D_MAX
    all_pred_ts.append(real_preds)
    # all_pred_ts.append(audio_ft * 10)


def translate_rms(wav):
    """
    Obtain RMS values for both voice channels.
    """
    sr = 16000
    rms_0 = librosa.feature.rms(y=wav[0], frame_length=int((params["w_size"] / 1000) * sr),
                              hop_length=int((params['f_shift'] / 1000) * sr), center=False)
    rms_1 = librosa.feature.rms(y=wav[1], frame_length=int((params["w_size"] / 1000) * sr),
                                hop_length=int((params['f_shift'] / 1000) * sr), center=False)
    return np.vstack([rms_0, rms_1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=str, default="test_20")
    parser.add_argument("--convo", type=str, default="None")
    parser.add_argument("--pred_csv", type=str, default="predictions.csv")
    parser.add_argument("--add_wav", action="store_true")
    parser.add_argument("--add_pred", action="store_true")
    parser.add_argument("--rms_thresh", type=float, default=0.01)
    parser.add_argument("--add_zero_diff", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    EXP_DIR = EXP_DIR + str("SILENCE_BASELINE") + "/"
    RMS_THRESH = args.rms_thresh

    include_wav = args.add_wav

    """
        Determine model type
    """

    fid = args.fid
    conv = args.convo
    if include_wav or args.add_pred:
        plot_dataset = SWDAGMMDataset([], f_id=fid, conv=conv, debug=True, d_max=D_MAX, st_size_limit=5, mode=2)
    else:
        plot_dataset = SWDAGMMDataset([], f_id=fid, conv=conv, debug=True, d_max=D_MAX, mode=2)

    fb = FeedbackInserter(EXP_DIR)
    # mc = MetricsCalculator(D_MAX, D_MAX_PRED)
    # zmc = ZeroDiffCalculator(D_MAX_PRED, 0.5)
    mc_ttee = MAECalculator(D_MAX)

    for i in range(len(plot_dataset.wavs)):
        if include_wav:
            fb.add_to_wav(plot_dataset.wavs[i])
        plot_dataset.wavs[i] = translate_rms(plot_dataset.wavs[i])

    kwargs = {'num_workers': 0, 'pin_memory': False} if device == 'cuda:0' else {}  # needed for using datasets on gpu
    plot_loader = torch.utils.data.DataLoader(plot_dataset, batch_size=20, shuffle=False,
                                              **kwargs)

    pred_start = []
    real_start = []
    for (d, labels) in plot_loader:
        if include_wav:
            fb.edit_wavs(labels['w_inds'], labels['chanl'], labels['beg'], labels['end'])
        predict_and_save(d, labels['start_ts'], labels['gmm_w'], labels['true_start_ts'])

    all_st_ts = np.vstack(all_st_ts)
    all_pred_ts = np.vstack(all_pred_ts)
    all_w_vecs = np.vstack(all_w_vecs)
    all_true_st_ts = np.vstack(all_true_st_ts)
    if include_wav:
        fb.add_bc(all_pred_ts)
    else:
        if args.add_pred:
            save_df(all_true_st_ts, all_pred_ts)
            df_p.to_csv(EXP_DIR + args.pred_csv, index=False)
        else:
            res = 16
            threshs = [x / res for x in range(0, D_MAX * res + 1, 1)]
            mets = mc_ttee.calc_ttees(all_true_st_ts, all_pred_ts, all_w_vecs, threshs)
            json.dump(mets, open(EXP_DIR + args.pred_csv.replace(".csv", "_ttee.json"), "w+"))


