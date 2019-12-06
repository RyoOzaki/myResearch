import numpy as np

def downsampling(signal, source_fs, target_fs):
    steps = source_fs // target_fs
    return signal[::steps]

def convert2mono(signal, channel_axis=1):
    if signal.ndim == 1:
        return signal
    else:
        return np.mean(signal, axis=channel_axis)
