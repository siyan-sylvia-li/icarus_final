import librosa
import numpy as np
from icarus_util import translate_index
import soundfile as sf
import shutil


class FeedbackInserter:
    def __init__(self, exp_dir):
        self.wavs = []
        self.wav_chunks = []
        self.uh_huh, sr = librosa.load("uhhuh.wav", sr=16000, mono=True)
        self.exp_dir = exp_dir

    def add_to_wav(self, wav):
        self.wavs.append(wav)

    def edit_wavs(self, w_inds, chanl, beg, end):
        for i in range(len(w_inds)):
            w_ind, c, b, e = int(w_inds[i].item()), \
                                     int(chanl[i].item()), \
                                     int(beg[i].item()), \
                                     int(end[i].item())
            b = translate_index(b)
            e = translate_index(e)
            print(w_ind, c, b, e)
            self.wav_chunks.append(self.wavs[w_ind][c, b:e])

    def translate_uhuh_inds(self, preds):
        """
            Insert backchannel at the actual time
        """
        actual_inds = np.zeros(preds.shape)
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                if preds[i, j] < 0.5:
                    offset = int(preds[i, j] / 0.05) # frame shift of 50 ms, calculate how many frames to shift to insert properly
                    actual_inds[i, j + offset] = 1
        return actual_inds

    def add_bc(self, preds):
        inds = self.translate_uhuh_inds(preds)
        for i in range(preds.shape[0]):
            wav_to_write = self.wav_chunks[i]
            new_channel = np.zeros(wav_to_write.shape)
            for idx in range(inds.shape[1]):
                if inds[i, idx] and inds[i, idx - 1] == 0:
                    idx = translate_index(idx)
                    if np.sum(new_channel[idx: idx + len(self.uh_huh)]) == 0:
                        print('WRITING')
                        new_channel[idx: idx + len(self.uh_huh)] = self.uh_huh[:min(len(new_channel)-idx, len(self.uh_huh))]
            print(sum(new_channel))
            print(sum(inds))
            wav_combine = (np.vstack([wav_to_write, new_channel])).transpose()
            sf.write(str(i) + "_combine.wav", wav_combine, samplerate=16000)
            shutil.move(str(i) + "_combine.wav", self.exp_dir + str(i) + "_combine.wav")

