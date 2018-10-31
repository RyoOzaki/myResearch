import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyhsmm.util.general import rle
from tqdm import tqdm

def unpack_durations(dur):
    unpacked = np.zeros(dur.sum())
    d = np.cumsum(dur[:-1])
    unpacked[d-1] = 1.0
    return unpacked

def _plot_raw_data(label_switching_data, title, raw_datas, cmap = None, **plotopts):
    if cmap is None:
        cmap = cm.binary
    ax = plt.subplot2grid((10, 1), (1, 0))
    plt.sca(ax)
    ax.matshow([label_switching_data], aspect = 'auto', cmap=cmap)
    plt.ylabel('Label switching')
    #label matrix
    ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
    plt.suptitle(title)
    plt.sca(ax)
    plt.plot(raw_datas, **plotopts)
    #write x&y label
    plt.xlabel('Frame')
    plt.ylabel('Value of feature')
    plt.xticks(())

source_data = "feature/concat_data.npz"
label_data = "feature/phoneme_label.npz"

npz_datas = np.load(source_data)
npz_labels = np.load(label_data)

for raw_name, raw_data in tqdm(npz_datas.items()):
    plt.clf()
    _, label_dur = rle(npz_labels[raw_name])
    unpacked_label_dur = unpack_durations(label_dur)
    _plot_raw_data(unpacked_label_dur, raw_name, raw_data)
    plt.savefig("figure/{}.pdf".format(raw_name))

print("Done!")
