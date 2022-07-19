import numpy as np


def split_data(data):
    # Segment into 30 second chunks
    interval_time = 480000
    add_time = 160

    def divide(data):
        total_data = []
        for i in range(0, data.shape[-1], interval_time):
            total_data.append(data[:, max(i - add_time, 0): min(i + interval_time + add_time, data.shape[-1])])
        return total_data

    if len(data.shape) == 1:
        data = data.unsqueeze(0)
    total = divide(data)
    return total


def translate_index(ind, sr=16000, f_s=0.05):
    ind = int(ind * f_s * sr)
    return ind