import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from itertools import product

parser = ArgumentParser()

parser.add_argument("--confusion_matrix", type=Path, required=True)
parser.add_argument("--print_iter", type=int, default=-1)

args = parser.parse_args()
conf_npy = np.load(args.confusion_matrix).astype(int)
print(conf_npy.shape)
if conf_npy.ndim == 4:
    conf_matrix = conf_npy[:, args.print_iter]
    speaker_num = conf_matrix.shape[0]
    truth_num = conf_matrix.shape[1]
    pred_num = conf_matrix.shape[2]

    # fig, ax = plt.subplots(ncols=speaker_num)
    fig, ax = plt.subplots(nrows=speaker_num)

    for spk_idx in range(speaker_num):
        ax[spk_idx].imshow(conf_matrix[spk_idx])
        ax[spk_idx].set_yticks(np.arange(truth_num))
        ax[spk_idx].set_xticks(np.arange(pred_num))
        ax[spk_idx].set_yticklabels(np.arange(truth_num))
        ax[spk_idx].set_xticklabels(np.arange(pred_num))
        ax[spk_idx].set_ylabel("Truth index")
        ax[spk_idx].set_xlabel("Predict index")
        for i, j in product(range(pred_num), range(truth_num)):
            if conf_matrix[spk_idx, j, i] != 0:
                ax[spk_idx].text(i, j, conf_matrix[spk_idx, j, i], ha="center", va="center", color="white")
else:
    conf_matrix = conf_npy[args.print_iter]
    truth_num = conf_matrix.shape[0]
    pred_num = conf_matrix.shape[1]

    # fig, ax = plt.subplots(ncols=speaker_num)
    fig, ax = plt.subplots()

    ax.imshow(conf_matrix)
    ax.set_yticks(np.arange(truth_num))
    ax.set_xticks(np.arange(pred_num))
    ax.set_yticklabels(np.arange(truth_num))
    ax.set_xticklabels(np.arange(pred_num))
    ax.set_ylabel("Truth index")
    ax.set_xlabel("Predict index")
    for i, j in product(range(pred_num), range(truth_num)):
        if conf_matrix[j, i] != 0:
            ax.text(i, j, conf_matrix[j, i], ha="center", va="center", color="white")
plt.show()
