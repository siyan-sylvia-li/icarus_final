import glob
import json
import os.path
import random
import threading

import numpy
import pickle

import pandas
import argparse
import librosa
import h5py

from multiprocessing import Pool

import time
import datetime

from align_transcript import clean_num, clean_no_split
import re

"""
    Global list of parameters
"""
params = {"w_size": 100, "f_shift": 50, "utt_len": 1200}
acts_dict = [x[:-1] for x in open("dialogue_acts.txt", "r").readlines()]
randomize = True
f_id = ""
wrong_sound_files = set()

random.seed(42)

storage = []
storage_lock = threading.Lock()
len_conv = 0

f = None


def extract_only_turn(utt_ts_df):
    prev_caller = [""]
    for i, row in utt_ts_df.iterrows():
        prev_caller.append(row['caller'])
    prev_caller.pop(-1)
    utt_ts_df['prev_caller'] = prev_caller
    utt_ts_df['diff_caller'] = numpy.where(utt_ts_df['prev_caller'] != utt_ts_df['caller'], True, False)
    utt_ts_df['ts_valid'] = numpy.where(utt_ts_df['end_ts'] > utt_ts_df['start_ts'], True, False)
    print(utt_ts_df)
    return utt_ts_df


def extract_by_speaker(conv):
    wav_buffer = []
    wav_file = glob.glob(conv + "/*.wav")[0]
    # TODO: Change to Librosa since that allows re-sampling
    wav, s_r = librosa.load(wav_file, sr=16000, mono=False)
    # wav, s_r = torchaudio.load(wav_file)
    """
        If this new conversation is in the list of convos with problem,
        then we switch the two channels.
    """
    if conv in wrong_sound_files:
        wav_buffer.append((wav[0], s_r))  # for A
        # Do the voice channel flip
        wav_buffer.append((wav[1], s_r))  # for B
    else:
        # TODO: Store convo incorrect
        wav_buffer.append((wav[1], s_r))  # for A
        # Do the voice channel flip
        wav_buffer.append((wav[0], s_r))  # for B
    return wav_buffer


def save_data(data):
    global storage, storage_lock
    storage_lock.acquire()
    storage.append(data)
    print("Current storage length", len(storage))
    storage_lock.release()


def save_h5py(data):
    global f
    grp = f.create_group(data['conv'])
    grp.create_dataset("indices", data=data['indices'])
    grp.create_dataset("wavs", data=data['wavs'])
    grp.create_dataset("st_ts", data=data['st_ts'])
    grp.create_dataset("end_ts", data=data['end_ts'])
    grp.create_dataset("d_act", data=data['d_act'])
    grp.create_dataset("w_vec", data=data['w_vec'])
    grp.create_dataset("words", data=data['words'])
    # print(data['transcript'])

    len_trans = max(len(data['transcript'][0]), len(data['transcript'][1]))
    data['transcript'][0] = data['transcript'][0] + [""] * (len_trans - len(data['transcript'][0]))
    data['transcript'][1] = data['transcript'][1] + [""] * (len_trans - len(data['transcript'][1]))
    dt = h5py.special_dtype(vlen=str)
    trans_set = grp.create_dataset("transcript", (2, len_trans), dtype=dt)
    trans_set[0] = data['transcript'][0]
    trans_set[1] = data['transcript'][1]
    print(grp['transcript'][0])


def check_and_save():
    global len_conv, storage, storage_lock
    storage_lock.acquire()
    if len(storage):
        data = storage.pop(0)
        storage_lock.release()
        save_h5py(data)
        len_conv = len_conv - 1
        print("Accessing file ... Current len_conv:", len_conv)
    else:
        storage_lock.release()


def saver_thread():
    global storage, len_conv, f
    while len_conv:
        check_and_save()

    f.close()


# def aud_feats_original(wav, sr):
#     global params
#     kaldi_pitch = torchaudio.functional.compute_kaldi_pitch(wav.unsqueeze(0), float(sr), params['w_size'],
#                                                             params['f_shift'])
#     kaldi_pitch = kaldi_pitch[:, :, 0]
#
#     mfcc = torchaudio.compliance.kaldi.mfcc(wav.unsqueeze(0), frame_length=params['w_size'],
#                                             frame_shift=params['f_shift'], sample_frequency=float(sr),
#                                             num_ceps=40, num_mel_bins=40, channel=0)
#     rms = librosa.feature.rms(y=wav, frame_length=int((params["w_size"] / 1000) * sr),
#                               hop_length=int((params['f_shift'] / 1000) * sr), center=False)
#     rms = rms.reshape((rms.shape[1], -1))
#     rms = torch.from_numpy(rms)
#     aud_fts = [torch.cat((mfcc[j, :], kaldi_pitch[:, j], rms[j, :]), dim=0).unsqueeze(0) for j in
#                range(kaldi_pitch.shape[1])]
#     return aud_fts, kaldi_pitch.shape[1]


# def aud_feats_pitch_freq(wav, sr):
#     global params
#
#     return aud_fts, kaldi_pitch.shape[1]


def calc_weight_vector(start_ts):
    """
        Calculate the weight vector based on start timestamp
        So that we can properly calculate the losses
    """
    # 1 / (1 + target)
    cumulative_zeros = 0
    zero_threshold = 10
    w_vec = numpy.zeros(start_ts.shape)

    for i in range(start_ts.shape[-1]):
        if start_ts[i] == 0:
            cumulative_zeros += 1
            if cumulative_zeros > zero_threshold:
                continue
        else:
            cumulative_zeros = 0
        w_vec[i] = 1 / (start_ts[i] + 1)
    return w_vec


def aud_feats_none(wav, sr):
    """
        Because we are extracting wav2vec features on the fly,
        we are just extracting audio segments here without processing.
    """
    # global params
    # step_size = int((params['w_size'] / 1000) / (1 / sr))
    # shift_size = int((params['f_shift'] / 1000) / (1 / sr))
    # result = get_windows(wav, step_size, shift_size)
    total_frames = (int(len(wav) / (sr / 100)) - 2) // 5
    return wav, total_frames


"""
    Create dictionary from mrk files
"""


def gen_dict_from_mrk(conv):
    mrk_time = [[], []]
    mrk_file = glob.glob(conv + "/*.mrk")[0]
    mrk_lines = open(mrk_file, "r").readlines()
    transcript = [[], []]
    for i in range(len(mrk_lines)):
        mrk_lines[i] = re.sub(r' +', '|', mrk_lines[i]).replace("\n", "")
        mrk_lines[i] = [x for x in mrk_lines[i].split('|') if len(x)]
        if len(mrk_lines[i]):
            mrk_lines[i][-1] = clean_no_split(mrk_lines[i][-1])
            mrk_lines[i][1] = clean_num(mrk_lines[i][1])
            mrk_lines[i][2] = clean_num(mrk_lines[i][2])
            try:
                # Compute end times of each word
                x = float(mrk_lines[i][1]) + float(mrk_lines[i][2])
                if "A" in mrk_lines[i][0]:
                    mrk_time[1].append(x)
                    transcript[1].append(mrk_lines[i][-1])
                elif "B" in mrk_lines[i][0]:
                    mrk_time[0].append(x)
                    transcript[0].append(mrk_lines[i][-1])
            except ValueError:
                continue
    assert len(transcript[0]) == len(mrk_time[0])
    assert len(transcript[1]) == len(mrk_time[1])
    print("Finished Processing ->", conv)
    print("MRK TIME 0", mrk_time[0])
    return mrk_time, transcript


def extract_data(conv):
    global params, acts_dict, randomize, f_id, storage
    while len(storage) >= 3:
        print("Waiting for storage to clean up")
    print(conv)
    wav_buffer = extract_by_speaker(conv)
    mrk_time, transcript = gen_dict_from_mrk(conv)
    utt_ts_df = pandas.read_csv(glob.glob(conv + "/*.ts.csv")[0])
    utt_ts_df = extract_only_turn(utt_ts_df)
    # Predict the second speaker's utterance
    all_d_acts = [[], []]
    all_start_ts = [[], []]
    all_end_ts = [[], []]
    all_audio_fts = [[], []]
    all_words = [[], []]
    for i, c in enumerate(['A', 'B']):
        wav, sr = wav_buffer[i]
        print(wav.shape, sr, "WAV DATA")
        pred_utt = utt_ts_df[((utt_ts_df['caller'] == c) & (utt_ts_df['ts_valid']))]
        aud_fts, total_time = aud_feats_none(wav, sr)

        curr_timestamp = 0
        curr_uind = 0

        all_audio_fts[i].extend(aud_fts)
        del aud_fts

        for k in range(total_time):
            curr_utt = pred_utt.iloc[curr_uind]
            if curr_timestamp >= curr_utt['end_ts']:
                curr_uind += 1
                try:
                    curr_utt = pred_utt.iloc[curr_uind]
                except IndexError:
                    print("pred_utt len: ", len(pred_utt))
                    print("Can't locate index", curr_uind)
                    break
            if curr_utt['start_ts'] >= curr_timestamp and curr_utt['diff_caller']:
                offset_start = curr_utt['start_ts'] - curr_timestamp
                offset_end = curr_utt['end_ts'] - curr_timestamp
            else:
                # during an utterance
                offset_start = 0.0
                offset_end = 0.0
            all_start_ts[i].append(offset_start)
            all_end_ts[i].append(offset_end)
            all_d_acts[i].append(
                acts_dict.index(curr_utt['act_tag']) if curr_utt['act_tag'] in acts_dict else len(acts_dict))
            mrk_time_filter = [ind for ind in range(len(mrk_time[i])) if mrk_time[i][ind] < curr_timestamp]
            all_words[i].append(
                mrk_time_filter[-1] if len(mrk_time_filter) else -1
            )
            curr_timestamp = curr_timestamp + params['f_shift'] / 1000

    total_windows = int((min(len(all_start_ts[1]), len(all_start_ts[0])) - params['utt_len']) * 0.04) if randomize else min(
        len(all_start_ts[0]), len(all_start_ts[1])) - params['utt_len']

    counter = 0
    ind_20 = 0 # 60 second chunks every 20 seconds, so give increment of 20
    inc_20 = params['utt_len'] // 3
    indices = []
    st_ts = []
    end_ts = []
    d_act = []
    w_vecs = []
    words = []
    while counter < total_windows and ind_20 < min(len(all_start_ts[0]), len(all_start_ts[1])) - params['utt_len']:
        channel = random.randint(0, 1)
        # Randomly generate an index if randomize
        if randomize:
            ind = random.randint(0, len(all_start_ts[channel]) - params['utt_len'] - 1)
        else:
            ind = ind_20
        # Access the start time, make sure the start time is around the next 5 seconds
        st_time = all_start_ts[channel][ind]
        if (5 <= st_time < 60 and randomize) or (not randomize):
            # print(all_audio_fts[rand_ind: rand_ind + params['utt_len']])
            curr_sample = numpy.array([channel, ind, ind + params['utt_len']])
            curr_st_ts = numpy.array(all_start_ts[channel][ind: ind + params['utt_len']])
            curr_end_ts = numpy.array(all_end_ts[channel][ind: ind + params['utt_len']])
            curr_d_act = numpy.array(all_d_acts[channel][ind: ind + params['utt_len']])
            indices.append(curr_sample)
            st_ts.append(curr_st_ts)
            end_ts.append(curr_end_ts)
            d_act.append(curr_d_act)
            words.append(all_words[channel][ind: ind + params['utt_len']])
            w_vecs.append(calc_weight_vector(curr_st_ts))
            counter = counter + 1

            """
                Increment of 20 seconds
            """
            ind_20 = ind_20 + inc_20
    indices = numpy.vstack(indices)
    print("Extracted Audio Feature Shape ->", indices.shape)
    st_ts = numpy.vstack(st_ts)
    end_ts = numpy.vstack(end_ts)
    d_act = numpy.vstack(d_act)
    w_vecs = numpy.vstack(w_vecs)
    return {
        "conv": conv.replace("full_data/", ""),
        "indices": indices,
        "wavs": all_audio_fts,
        "transcript": transcript,
        "st_ts": st_ts,
        "end_ts": end_ts,
        "d_act": d_act,
        "w_vec": w_vecs,
        "words": words
    }


def check_wrong(wrong_list, c):
    for w in wrong_list:
        if w in c:
            return True
    return False


if __name__ == "__main__":
    print(datetime.datetime.now())
    start_time = time.time()
    # Check if split json file exists
    if not os.path.exists("train_val_test_split.json"):
        complete_conv_list = glob.glob("full_data/*")
        full_conv_list = []
        wrong_list = pickle.load(open("wrong_files.p", "rb"))
        for c in complete_conv_list:
            if not check_wrong(wrong_list, c) and len(glob.glob(c + "/*.wav")):
                full_conv_list.append(c)

        train_list = random.sample(full_conv_list, k=int(0.8 * len(full_conv_list)))
        eval_test_list = [x for x in full_conv_list if x not in train_list]
        eval_list = random.sample(eval_test_list, k=int(0.5 * len(eval_test_list)))
        test_list = [x for x in eval_test_list if x not in eval_list]
        with open("train_val_test_split.json", "w+") as f:
            json.dump({"train": train_list, "val": eval_list, "test": test_list}, f)
        # Train on 80% of convo, eval on 10%, test on 10%

    # Generate on train-test split
    split_data = json.load(open('train_val_test_split.json'))
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str)
    parser.add_argument("--size", type=int, default=-1)
    parser.add_argument("--seq", action="store_true")
    args = parser.parse_args()

    wrong_sound_files = set([x.replace("\n", "") for x in open("wrong_sound_files.txt", "r").readlines()])

    conv_list = []
    if args.part == "train":
        conv_list = split_data['train']
        f_id = "train"
    elif args.part == "test":
        conv_list = split_data['test']
        f_id = 'test'
        randomize = False
    elif args.part == "val":
        conv_list = split_data['val']
        f_id = "val"
        randomize = False

    if args.size != -1:
        conv_list = conv_list[: args.size]
        f_id = f_id + "_" + str(args.size)

    f = h5py.File("processed_data/" + f_id + ".hdf5", "a")

    len_conv = len(conv_list)

    if args.seq:
        t = threading.Thread(target=saver_thread)
        t.start()
        for c in conv_list:
            data = extract_data(c)
            save_data(data)
        t.join()
    else:
        t = threading.Thread(target=saver_thread)
        t.start()
        with Pool(processes=min(len(conv_list), 5)) as pool:
            for c in conv_list:
                pool.apply_async(extract_data, (c,), callback=save_data)
            # pool.map(extract_data, conv_list)

            pool.close()
            pool.join()
        t.join()

    # Generate plotting test dialogue data
    # plot_dialogue = ["full_data/sw4342"]
    # plot_dataset = SWDADataset(plot_dialogue, "sw4342", randomize=False)
    # plot_dataset.save_data("sw4342")
    print(datetime.datetime.now())
    print("TOTAL ELAPSED -> %s" % (time.time() - start_time))
