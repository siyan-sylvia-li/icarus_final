import time

from swda_gmm_dataset import SWDAGMMDataset
import os
import pandas
import argparse

from icarus_heatmap import ModeGPTLSTMHeatmap
from icarus_gmm_trainer import get_wav2vec_rms

from icarus_util import split_data

import torch
import fairseq
from meticulous import Experiment

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(42)

"""
    Loading wav2vec model
"""
cp_path = 'wav2vec/wav2vec_large.pt'
wav2vec_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
wav2vec_model = wav2vec_model[0]
wav2vec_model.to(device)
wav2vec_model.eval()

df = None
da_ce = torch.nn.CrossEntropyLoss(reduction='none')

"""
    Param Sweep Functionality
"""
col_params = ['batch_size', 'lr']
# Template for each individual run
df_run = {}


def calc_loss(output, target, ws):
    loss = torch.sum(torch.square((output - target)), dim=-1)
    loss = torch.mean(loss * ws)
    return loss


def calc_da_loss(output, target, ws):
    output = torch.transpose(output, 1, 2)
    loss = da_ce(output, target)
    loss = torch.mean(loss * ws)
    return loss


def pass_wav2vec(data):
    data = data.to(device)
    if len(data.shape) == 1:
        data = data.unsqueeze(0)
    batch_size = 32
    results = []
    with torch.no_grad():
        for i in range(0, data.shape[0], batch_size):
            z = wav2vec_model.feature_extractor(data[i: min(i + batch_size, data.shape[0]), :])
            c = wav2vec_model.feature_aggregator(z)
            del z
            results.append(c)
            torch.cuda.empty_cache()
    if len(results) > 1:
        results = torch.cat(results, dim=0)
    else:
        results = results[0]
    return results


def obtain_wav2vec(data):
    total = split_data(data)
    with torch.no_grad():
        wavs = [pass_wav2vec(w) for w in total]
    wav = torch.cat(wavs, dim=2)
    """
        Only obtain every 5 frames to maintain 50 ms step size
    """
    wav = torch.index_select(wav, 2, torch.tensor([5 * i for i in range(int(wav.shape[-1] / 5))]).to(device))
    wav = torch.transpose(wav, 1, 2)
    return wav


def train(model, train_loader, epoch):
    global df, col_params, add_std_loss, add_consistency_loss
    model.train()
    # Compute norms
    # Initiate running losses
    run_st_ts_loss = 0
    for id, (data, labels) in enumerate(train_loader):
        """
            Pass through the wav2vec model
        """
        optimizer.zero_grad()
        hidden = model.init_hidden(data.shape[0])
        labels['word_inds'] = labels['word_inds'].to(device)
        out = model(data, hidden, labels['transcript'], labels['word_inds'])
        st_ts_loss = calc_loss(out, labels['g'].to(device), labels['gmm_w'].to(device))
        run_st_ts_loss += st_ts_loss.detach().item()

        losses = st_ts_loss # Removing end ts loss and d_act
        losses.backward()

        optimizer.step()

    run_st_ts_loss = run_st_ts_loss / len(train_loader)

    # if id % 100 == 0:
    print("Start TS Loss = {l1}".format(
        l1=run_st_ts_loss))
    return (run_st_ts_loss, {"st_loss_avg_" + str(epoch): run_st_ts_loss})


def test(model, test_loader, epoch):
    model.eval()
    st_ts = 0
    for test_data, labels in test_loader:
        # test_data = obtain_wav2vec(test_data)
        labels['word_inds'] = labels['word_inds'].to(device)
        out = model(test_data, model.init_hidden(test_data.shape[0]), labels['transcript'], labels['word_inds'])
        st_ts += calc_loss(out, labels['g'].to(device), labels['gmm_w'].to(device)).item()
    st_ts = st_ts / len(test_loader)
    print('Eval Losses: Start, Start MSE => ', st_ts)
    return st_ts, {"st_loss_eval_" + str(epoch): st_ts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--message", type=str)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=int, default=0, help="0 if GPT2 + wav2vec, 1 if wav2vec only, 2 if GPT2 only")
    parser.add_argument("--zero_thresh", type=int, default=40)
    parser.add_argument("--dmax", type=int, default=5)
    parser.add_argument("--adj", type=int, help="Adjustment mode for SWDA GMM Dataset", default=2)
    parser.add_argument("--resolution", type=int, default=4)

    Experiment.add_argument_group(parser)
    experiment = Experiment.from_parser(parser)
    args = parser.parse_args()

    experiment.metadata.update({"Purpose Message": args.message})
    d_max = args.dmax
    res = args.resolution

    if args.debug:
        size_limit = 3
    else:
        size_limit = -1
    train_set = SWDAGMMDataset([], f_id="train_200", debug=args.debug, size_limit=size_limit, d_max=d_max,
                               mode=args.adj, resolution=res)
    val_set = SWDAGMMDataset([], f_id="val_20", debug=args.debug, size_limit=size_limit, d_max=d_max, mode=args.adj,
                             resolution=res)

    """
        Use wav2vec to obtain embeddings on the fly
    """
    if args.mode in [0, 1, 3]:
        for i in range(len(train_set.wavs)):
            if args.mode == 3:
                train_set.wavs[i] = get_wav2vec_rms(train_set.wavs[i])
            else:
                train_set.wavs[i] = obtain_wav2vec(train_set.wavs[i])
        for i in range(len(val_set.wavs)):
            if args.mode == 3:
                val_set.wavs[i] = get_wav2vec_rms(val_set.wavs[i])
            else:
                val_set.wavs[i] = obtain_wav2vec(val_set.wavs[i])
    del wav2vec_model
    torch.cuda.empty_cache()

    hyperparams = {'n_feature': train_set.get_feats_size()}
    kwargs = {'num_workers': 0, 'pin_memory': False} if device == 'cuda:0' else {}  # needed for using datasets on gpu

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               **kwargs)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True,
                                              **kwargs)

    # Check if tsv file exists, if not create new file
    # File based on batch size
    df_run = {"lr": args.lr, "batch_size": args.batch_size}
    print(df_run)
    # if not os.path.exists(args.param_path):
    #     df = pandas.DataFrame(columns=col_params)
    # else:
    #     df = pandas.read_csv(args.param_path)

    if args.model is None:
        model = ModeGPTLSTMHeatmap(hyperparams['n_feature'], mode=args.mode, T=d_max * 2 * res)
    else:
        model = torch.load(args.model)
    model = model.to(device)
    model.float()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    clip = 5  # gradient clipping

    best_loss = 10e3
    best_tr_loss = 10e3
    for epoch in range(1, args.epoch + 1):
        start_time = time.time()
        df = pandas.DataFrame(columns=col_params)
        print("Epoch {s} ============".format(s=epoch))
        # scheduler.step()
        tr_loss, tr_data = train(model, train_loader, epoch)
        ev_loss, eval_data = test(model, val_loader, epoch)
        df_run.update(tr_data)
        df_run.update(eval_data)
        if ev_loss < best_loss:
            best_loss = ev_loss
            torch.save(model, experiment.curexpdir + "/best-loss-model.pt")
            experiment.summary({'eval_loss': ev_loss, 'current_epoch': epoch})
        if tr_loss < best_tr_loss:
            best_tr_loss = tr_loss
            torch.save(model, experiment.curexpdir + "/train-loss-model.pt")
        if epoch % 1 == 0:
            torch.save(model, experiment.curexpdir + "/epoch-model-" + str(epoch) + ".pt")
        print("Epoch Took:", time.time() - start_time)
        if os.path.exists(experiment.curexpdir + "/run_stats.csv"):
            os.remove(experiment.curexpdir + "/run_stats.csv")
        df = df.append(df_run, ignore_index=True)
        df.to_csv(experiment.curexpdir + "/run_stats.csv", index=False)
    torch.save(model, experiment.curexpdir + "/model-after-train.pt")
