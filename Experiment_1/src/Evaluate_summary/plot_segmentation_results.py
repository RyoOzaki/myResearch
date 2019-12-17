import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import japanize_matplotlib

from configparser import ConfigParser

class ConfigParser_with_eval(ConfigParser):
    def get(self, *argv, **kwargs):
        import numpy
        val = super(ConfigParser_with_eval, self).get(*argv, **kwargs)
        return eval(val)

def load_config(filename):
    cp = ConfigParser_with_eval()
    cp.read(filename)
    return cp

def _boundary(label):
    diff = np.diff(label, axis=1)
    diff[diff!=0] = 1
    zeros =  np.zeros((diff.shape[0], 1))
    return np.concatenate((diff,zeros), axis=1)

def mixed_label(label, Ft, clist):
    label = np.asarray(label, dtype=int)
    Ft = np.asarray(Ft, dtype=int)
    if label.ndim == 1:
        label = label.reshape((1, label.shape[0]))
    if Ft.ndim == 1:
        Ft = Ft.reshape((1, Ft.shape[0]))
    colors = [ [ clist[idx] for idx in row_label ] for row_label in label ]
    for Ftx, Fty in zip(*np.where(Ft == 1)):
        colors[Ftx][Fty] = (0, 0, 0, 1)
    return colors


def plot_result(fig, phn_clist, wrd_clist, phn_label, wrd_label, phn_Ft, wrd_Ft, letter_stateseq, word_stateseq, word_durations):
    # phn truth
    ax = plt.subplot2grid((10, 2), (0, 0), fig=fig)
    # plt.sca(ax)
    mixed = mixed_label(phn_label, phn_Ft, phn_clist)
    ax.matshow(mixed, aspect='auto')
    ax.set_title("Phoneme result")
    ax.set_xticks([])
    ax.set_yticks([])

    # wrd truth
    ax = plt.subplot2grid((10, 2), (0, 1), fig=fig)
    # plt.sca(ax)
    mixed = mixed_label(wrd_label, wrd_Ft, wrd_clist)
    ax.matshow(mixed, aspect='auto')
    ax.set_title("Word result")
    ax.set_xticks([])
    ax.set_yticks([])

    # phn result
    ax = plt.subplot2grid((10, 2), (1, 0), rowspan=9, fig=fig)
    # plt.sca(ax)
    mixed = mixed_label(letter_stateseq, _boundary(letter_stateseq), phn_clist)
    ax.matshow(mixed, aspect='auto')
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Iterations")

    # wrd result
    ax = plt.subplot2grid((10, 2), (1, 1), rowspan=9, fig=fig)
    # plt.sca(ax)
    mixed = mixed_label(word_stateseq, word_durations, wrd_clist)
    ax.matshow(mixed, aspect='auto')
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Iterations")

default_hypparams_model = Path("hypparams/model.config")

parser = ArgumentParser()

parser.add_argument("--model", type=Path, default=default_hypparams_model, help=f"hyper parameters of model, default is [{default_hypparams_model}]")

parser.add_argument("--result_dir", type=Path)
parser.add_argument("--word_stateseq", type=Path)
parser.add_argument("--letter_stateseq", type=Path)
parser.add_argument("--word_durations", type=Path)

parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--wrd", type=Path, required=True)

parser.add_argument("--Ft_phn", type=Path, required=True)
parser.add_argument("--Ft_wrd", type=Path, required=True)

parser.add_argument("--figsize", type=int, nargs=2, default=[16, 8])

args = parser.parse_args()

#%% config parse
print("Loading model config...")
config_parser = load_config(args.model)
section = config_parser["model"]
word_num = section["word_num"]
letter_num = section["letter_num"]
phn_color_list = [cm.tab20(float(i)/letter_num) for i in range(letter_num)]
wrd_color_list = [cm.tab20(float(i)/word_num) for i in range(word_num)]
print("Done!")

if args.result_dir:
    result_dir = args.result_dir
    word_stateseq_file = result_dir / "word_stateseq.npz"
    letter_stateseq_file = result_dir / "letter_stateseq.npz"
    word_durations_file = result_dir / "word_durations.npz"
else:
    word_stateseq_file = args.word_stateseq
    letter_stateseq_file = args.letter_stateseq
    word_durations_file = args.word_durations

assert word_stateseq_file is not None
assert letter_stateseq_file is not None
assert word_durations_file is not None

word_stateseq_npz = np.load(word_stateseq_file)
letter_stateseq_npz = np.load(letter_stateseq_file)
word_durations_npz = np.load(word_durations_file)

phn_label = np.load(args.phn)
wrd_label = np.load(args.wrd)
Ft_phn_label = np.load(args.Ft_phn)
Ft_wrd_label = np.load(args.Ft_wrd)

keys = sorted(list(word_stateseq_npz.keys()))


for key in keys:

    fig = plt.figure(figsize=args.figsize)

    phnlab = phn_label[key]
    wrdlab = wrd_label[key]
    phnFt = Ft_phn_label[key]
    wrdFt = Ft_wrd_label[key]
    lstsq = letter_stateseq_npz[key]
    wstsq = word_stateseq_npz[key]
    wdur = word_durations_npz[key]
    plot_result(fig, phn_color_list, wrd_color_list, phnlab, wrdlab, phnFt, wrdFt, lstsq, wstsq, wdur)
    plt.show()
