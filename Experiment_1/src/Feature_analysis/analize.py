import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from util.utility import separate_speaker, get_separated_values
import matplotlib.pyplot as plt
import itertools
from sklearn.decomposition import PCA
import copy

def plot_datas(data, ax, bins, color=None, hist_alpha=None, data_alpha=None, xlim=None, ylim=None):
    hist_color = None
    data_color = None
    if color is not None:
        color_tmp = list(color)

        hist_color = copy.copy(color_tmp)
        hist_color[-1] = hist_alpha
        hist_color = tuple(hist_color)

        data_color = copy.copy(color_tmp)
        data_color[-1] = data_alpha
        data_color = tuple(data_color)
    dim = data.shape[1]
    xlim_given = False
    ylim_given = False
    x_minmax = [data.max(), data.min()]
    y_minmax = [data.max(), data.min()]
    if xlim is not None:
        xlim_given = True
        x_minmax = xlim
    if ylim is not None:
        ylim_given = True
        y_minmax = ylim
    for x, y in itertools.product(range(dim), repeat=2):
        axes = ax[y, x]
        if x == y:
            axes.hist(data[:, x], bins=bins, color=hist_color)
        else:
            axes.scatter(data[:, x], data[:, y], marker=".", color=data_color)
            if not xlim_given and y < x:
                xlim = axes.get_xlim()
                x_minmax[0] = min(x_minmax[0], xlim[0])
                x_minmax[1] = max(x_minmax[1], xlim[1])
            if not ylim_given and y < x:
                ylim = axes.get_ylim()
                y_minmax[0] = min(y_minmax[0], ylim[0])
                y_minmax[1] = max(y_minmax[1], ylim[1])
    for x, y in itertools.product(range(dim), repeat=2):
        axes = ax[y, x]
        if x != y:
            if y < x:
                axes.set_xlim(*x_minmax)
                axes.set_ylim(*y_minmax)
            else:
                axes.set_xlim(*y_minmax)
                axes.set_ylim(*x_minmax)
        if y == dim - 1:
            axes.set_xlabel(f"Axes {x}")
        if x == 0:
            axes.set_ylabel(f"Axes {y}")

def get_xlim_ylim(datas, ax):
    data = np.concatenate(datas, axis=0)
    dim = data.shape[1]
    x_minmax = [data.max(), data.min()]
    y_minmax = [data.max(), data.min()]
    for y in range(dim-1):
        for x in range(y+1, dim):
            axes = ax[y, x]
            axes.scatter(data[:, x], data[:, y], marker=".")
            xlim = axes.get_xlim()
            ylim = axes.get_ylim()
            x_minmax[0] = min(x_minmax[0], xlim[0])
            x_minmax[1] = max(x_minmax[1], xlim[1])
            y_minmax[0] = min(y_minmax[0], ylim[0])
            y_minmax[1] = max(y_minmax[1], ylim[1])
    return x_minmax, y_minmax

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)
parser.add_argument("--phn", type=Path)

parser.add_argument("--cmap", type=str, default="tab10")
parser.add_argument("--bins", type=int)
parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8])
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--hist_alpha", type=float)
parser.add_argument("--data_alpha", type=float)

parser.add_argument("--mode", type=str, choices=["none", "phn", "spk"])

parser.add_argument("--savefig", type=Path)

parser.add_argument("--with_pca", action="store_true")
parser.add_argument("--n_components", type=int)

args = parser.parse_args()

if args.mode is not None:
    mode = args.mode
elif args.phn is not None and args.speaker_id is not None:
    mode = args.mode
    while mode not in ["phn", "spk", "none"]:
        mode = input("Which do you want plot (phn / spk / none) > ")
elif args.phn is not None:
    mode = "phn"
elif args.speaker_id is not None:
    mode = "spk"
else:
    mode = "none"

source = np.load(args.source)
keys = sorted(list(source.keys()))

hist_alpha = args.hist_alpha or args.alpha
data_alpha = args.data_alpha or args.alpha

if args.with_pca:
    packed_source, lengths = packing([source[key] for key in keys])
    dim = args.n_components or packed_source.shape[1]
    pca = PCA(n_components=args.n_components)
    pca.fit(packed_source)
    packed_source = pca.transform(packed_source)
    unpacked = unpacking(packed_source, lengths)
    source = {}
    for key, data in zip(keys, unpacked):
        source[key] = data

dim = source[keys[0]].shape[1]

fig, ax = plt.subplots(dim, dim, figsize=args.figsize)

if mode == "none":
    data = np.concatenate([source[key] for key in keys], axis=0)
    plot_datas(data, ax, args.bins, color=None)
else:
    if mode == "spk":
        all_speaker, speaker_individual_keys = separate_speaker(np.load(args.speaker_id))
        N = len(all_speaker)
        datas = [
            np.concatenate([source[key] for key in spk_key], axis=0)
            for spk_key in speaker_individual_keys
        ]
    elif mode == "phn":
        phns = np.load(args.phn)
        tmp_datas = np.concatenate([source[key] for key in keys], axis=0)
        tmp_phns = np.concatenate([phns[key] for key in keys], axis=0)
        N = int(tmp_phns.max() + 1)
        datas = [
            tmp_datas[tmp_phns == phn]
            for phn in range(N)
        ]
    xlim, ylim = get_xlim_ylim(datas, ax)
    plt.close()
    fig, ax = plt.subplots(dim, dim, figsize=args.figsize)
    cmap = plt.get_cmap(args.cmap)
    for i in range(N):
        if datas[i].shape[0] == 0:
            continue
        plot_datas(datas[i], ax, args.bins, color=cmap(i), hist_alpha=hist_alpha, data_alpha=data_alpha, xlim=xlim, ylim=ylim)

fig.suptitle(f"{args.source.stem}", fontsize=24)

if args.savefig is not None:
    plt.savefig(args.savefig)
else:
    plt.show()
