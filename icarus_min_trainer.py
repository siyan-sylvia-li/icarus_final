import time

from swda_gmm_dataset import SWDAGMMDataset
import os
import pandas
import argparse

from icarus_mode_gpt2_lstm import ModeGPTLSTM

from icarus_util import split_data

import torch
import fairseq
from meticulous import Experiment


def print_cuda_mem():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r - a  # free inside reserved
    print("Total memory ->", t, "Reserved ->", r, "Allocated ->", a, "Free ->", f)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

"""
    Loading wav2vec model
"""
cp_path = 'wav2vec/wav2vec_large.pt'
wav2vec_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
wav2vec_model = wav2vec_model[0]
wav2vec_model.to(device)
wav2vec_model.eval()

df = None

g_nll = torch.nn.GaussianNLLLoss(reduction='none')
st_ce = torch.nn.CrossEntropyLoss(reduction='none')
mse = torch.nn.MSELoss(reduction='none')
cross_entropy = torch.nn.CrossEntropyLoss()
"""
    Param Sweep Functionality
"""
col_params = ['batch_size', 'lr']
# Template for each individual run
df_run = {}
cap_std = -1
add_std_loss = False
add_consistency_loss = False
weighted = False
loss_mode = "gaussian"
alpha = 0.2
beta = 0.8


def calc_loss(output, target, w_vec=None):
    global cap_std, weighted, loss_mode
    new_output = torch.zeros(output.shape).to(device)
    new_output[:, :, 0] += output[:, :, 0]
    new_output[:, :, 1] += torch.clamp(output[:, :, 1], min=0.00001, max=None if cap_std < 0 else cap_std)
    # print(new_output[0, :, :].reshape(-1), target[0, :])
    if loss_mode == "gaussian":
        norm_prob = g_nll(new_output[:, :, 0], target, new_output[:, :, 1])
    elif loss_mode == "mse":
        norm_prob = mse(new_output[:, :, 0], target)
    else:
        # Weighted square error
        norm_prob = wse(new_output[:, :, 0], target)

    # Weight by proximity to the actual utterance
    # TODO: Precompute these weights and store in dataset?
    if weighted:
        if w_vec is None:
            weights = (torch.ones(norm_prob.shape).to(device) / (target + 1)).to(device)
        else:
            weights = w_vec.to(device)

        norm_prob = norm_prob * weights
    return torch.mean(norm_prob)


def calc_loss_plot(output, target):
    global cap_std
    new_output = torch.zeros(output.shape).to(device)
    new_output[:, :, 0] += output[:, :, 0]
    new_output[:, :, 1] += torch.clamp(output[:, :, 1], min=0.00001, max=None if cap_std < 0 else cap_std)
    # print(new_output[0, :, :].reshape(-1), target[0, :])
    norm_prob = g_nll(new_output[:, :, 0], target, new_output[:, :, 1])
    # Weight by proximity to the actual utterance
    if weighted:
        weights = (torch.ones(norm_prob.shape).to(device) / (target + 1)).to(device)
        norm_prob = norm_prob * weights
    return norm_prob


def bucket_ts_loss(output, target):
    output = torch.transpose(output, 1, 2)
    losses = st_ce(output, target)
    return torch.mean(losses)


def consistency_loss(output, target, w_vec):
    """
        Consistency loss, encourage model to produce time stamps
        with consistent offsets
    """
    output = output.to(torch.float)
    target = target.to(torch.float)
    offset_vect_out = output[:, 1:, 0] - output[:, :-1, 0]
    offset_vect_tar = target[:, 1:] - target[:, :-1]
    mse_off = mse(offset_vect_out.float(), offset_vect_tar.float())
    # print(mse_off, "AFTER")
    if weighted:
        mse_off = mse_off * w_vec[:, 1:]
    return torch.mean(mse_off)


def wse(output, target):
    global alpha, beta
    diff = output - target
    zeros = torch.zeros(diff.shape).to(device)
    wse = alpha * torch.square(torch.minimum(diff, zeros)) + beta * torch.square(torch.maximum(diff, zeros))
    return wse


def std_loss(output):
    """
        Punishes large predicted std values, regularization
    """
    output = output.to(torch.float)
    output_sum = torch.sum(torch.abs(output[:, :, 1]))
    return output_sum


def mse_loss(output, target, w_vec):
    output = output.to(torch.float)
    target = target.to(torch.float)
    mse_loss = mse(output[:, :, 0], target)
    if weighted:
        mse_loss = mse_loss * w_vec
    return torch.mean(mse_loss)


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
    # with torch.no_grad():
    #     z = wav2vec_model.feature_extractor(data)
    #     results = wav2vec_model.feature_aggregator(z)
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
    run_cons_loss = 0
    run_std_loss = 0
    for id, (data, labels) in enumerate(train_loader):
        """
            Pass through the wav2vec model
        """
        optimizer.zero_grad()
        # data = obtain_wav2vec(data)
        # print("DATA SHAPE", data.shape)
        hidden = model.init_hidden(data.shape[0])
        labels['word_inds'] = labels['word_inds'].to(device)
        out = model(data, hidden, labels['transcript'], labels['word_inds'])
        st_ts_loss = calc_loss(out, labels['start_ts'].to(device), w_vec=labels['gmm_w'].to(device))
        run_st_ts_loss += st_ts_loss.detach().item()

        losses = st_ts_loss # Removing end ts loss and d_act
        cons_loss = consistency_loss(out, labels['start_ts'].to(device), labels['gmm_w'].to(device)).float()
        run_cons_loss += cons_loss.detach().item()
        if add_consistency_loss:
            losses = losses + cons_loss
        if add_std_loss:
            std_value_loss = std_loss(out).float()
            run_std_loss += std_value_loss.detach().item()
            losses = losses + 0.1 * std_value_loss
        losses.backward()

        optimizer.step()

    run_st_ts_loss = run_st_ts_loss / len(train_loader)
    # Getting rid of end loss
    run_cons_loss = run_cons_loss / len(train_loader)
    run_std_loss = run_std_loss / len(train_loader)

    # if id % 100 == 0:
    print("Start TS Loss = {l1}, consistency loss = {l2}".format(
        l1=run_st_ts_loss, l2=run_cons_loss))
    return (run_st_ts_loss, {"st_loss_avg_" + str(epoch): run_st_ts_loss,
                             "std_loss_avg_" + str(epoch): run_std_loss
                             })


def test(model, test_loader, epoch):
    model.eval()
    st_ts = 0
    st_ts_mse = 0
    for test_data, labels in test_loader:
        # test_data = obtain_wav2vec(test_data)
        labels['word_inds'] = labels['word_inds'].to(device)
        out = model(test_data, model.init_hidden(test_data.shape[0]), labels['transcript'], labels['word_inds'])
        st_ts += calc_loss(out, labels['start_ts'].to(device), labels['gmm_w'].to(device)).item()
        if loss_mode == "gaussian":
            st_ts_mse += mse_loss(out, labels['start_ts'].to(device), labels['gmm_w'].to(device)).item()
    st_ts = st_ts / len(test_loader)
    st_ts_mse = st_ts_mse / len(test_loader)
    print('Eval Losses: Start, Start MSE => ', st_ts, st_ts_mse)
    return st_ts, {"st_loss_eval_" + str(epoch): st_ts, "st_loss_mse_eval_" + str(epoch): st_ts_mse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--drop_prob", type=float, default=0.1)
    # PARAM SWEEP FUNCTIONALITY
    parser.add_argument("--cap_std", type=float, default=-1)
    parser.add_argument("--add_consistency", action="store_true")
    parser.add_argument("--add_std", action="store_true", help="Whether to add penalization for large std values")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--mode", type=int, default=0, help="0 if GPT2 + wav2vec, 1 if wav2vec only, 2 if GPT2 only")
    parser.add_argument("--loss", type=str, default="gaussian")
    parser.add_argument("--zero_thresh", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.8)
    parser.add_argument("--adj", type=int, default=2)
    parser.add_argument("--d_max", type=int, default=2)

    Experiment.add_argument_group(parser)
    experiment = Experiment.from_parser(parser)
    args = parser.parse_args()

    add_consistency_loss = args.add_consistency
    add_std_loss = args.add_std
    weighted = args.weighted
    loss_mode = args.loss
    alpha = args.alpha
    beta = args.beta
    if args.cap_std > 0:
        cap_std = args.cap_std

    if args.debug:
        size_limit = 3
    else:
        size_limit = -1
    # train_set = SWDADataset([], f_id="train_200", weighted=weighted, debug=args.debug, size_limit=size_limit)
    # val_set = SWDADataset([], f_id="val_20", weighted=weighted, debug=args.debug, size_limit=size_limit)
    # test_set = SWDADataset([], f_id="test_20", weighted=weighted, debug=args.debug, size_limit=size_limit)
    train_set = SWDAGMMDataset([], f_id="train_200", debug=args.debug, size_limit=size_limit, d_max=args.d_max,
                               mode=args.adj)
    val_set = SWDAGMMDataset([], f_id="val_20", debug=args.debug, size_limit=size_limit, d_max=args.d_max, mode=args.adj)

    """
        Use wav2vec to obtain embeddings on the fly
    """
    if args.mode in [0, 1]:
        for i in range(len(train_set.wavs)):
            train_set.wavs[i] = obtain_wav2vec(train_set.wavs[i])
        for i in range(len(val_set.wavs)):
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
        model = ModeGPTLSTM(hyperparams['n_feature'], mode=args.mode)
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
        torch.save(model, experiment.curexpdir + "/epoch-model-" + str(epoch) + ".pt")
        print("Epoch Took:", time.time() - start_time)
        if os.path.exists(experiment.curexpdir + "/run_stats.csv"):
            os.remove(experiment.curexpdir + "/run_stats.csv")
        df = df.append(df_run, ignore_index=True)
        df.to_csv(experiment.curexpdir + "/run_stats.csv", index=False)
    torch.save(model, experiment.curexpdir + "/model-after-train.pt")
