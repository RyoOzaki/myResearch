import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import copy
from util.utility import separate_speaker, get_separated_values
from sklearn.decomposition import PCA

# phn_label_list : ['s', 'a', 'e', 'i', 'o', 'u']
# wrd_label_list : ['s', 'aioi', 'ao', 'aue', 'ie', 'uo']
def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

def set_alpha(colors, alpha):
    new_colors = []
    for color in colors:
        tmp = copy.copy(list(color))
        tmp[-1] = alpha
        new_colors.append(tuple(tmp))
    return new_colors

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path, required=True)
parser.add_argument("--phn_list", type=str, nargs="+")
parser.add_argument("--not_show", type=str, nargs="*")
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--cmap", type=str, default="tab10")
parser.add_argument("--output_dir", type=Path)
parser.add_argument("--format", default="pdf")

parser.add_argument("--swap_second", action="store_true")

args = parser.parse_args()

if args.output_dir is None:
    output_dir = Path(f"./figure/{args.source.stem}")
else:
    output_dir = args.output_dir
output_dir.mkdir(exist_ok=True, parents=True)

source = np.load(args.source)
keys = sorted(list(source.keys()))

packed_source, lengths = packing([source[key] for key in keys])
pca = PCA(n_components=2)
pca.fit(packed_source)
packed_source = pca.transform(packed_source)
if args.swap_second:
    packed_source[:, 1] *= -1
unpacked = unpacking(packed_source, lengths)
source = {}
for key, data in zip(keys, unpacked):
    source[key] = data

plt.rcParams["font.size"] = 18

plt.figure(figsize=(9, 7))
# phn
phns = np.load(args.phn)
tmp_datas = np.concatenate([source[key] for key in keys], axis=0)
tmp_phns = np.concatenate([phns[key] for key in keys], axis=0)
if args.phn_list is None:
    N = int(tmp_phns.max() + 1)
    phn_list = [f"{i+1}-th phn" for i in range(N)]
else:
    N = len(args.phn_list)
    phn_list = args.phn_list
datas = [
    tmp_datas[tmp_phns == phn]
    for phn in range(N)
]
cmap = plt.get_cmap(args.cmap)
colors = set_alpha([cmap(i) for i in range(N)], args.alpha)
cmap_cnt = 0
for i in range(N):
    if datas[i].shape[0] == 0 or phn_list[i] in args.not_show:
        continue
    plt.plot(datas[i][:, 0], datas[i][:, 1], ".", color=colors[cmap_cnt], label=phn_list[i])
    cmap_cnt += 1
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.legend()
plt.savefig(output_dir / f"data_colord_by_phn.{args.format}", bbox_inches='tight', pad_inches=0)

# speaker
plt.clf()
all_speaker, speaker_individual_keys = separate_speaker(np.load(args.speaker_id))
N = len(all_speaker)
datas = [
    np.concatenate([source[key] for key in spk_key], axis=0)
    for spk_key in speaker_individual_keys
]
colors = set_alpha([cmap(i) for i in range(N)], args.alpha)
cmap_cnt = 0
for i in range(N):
    if datas[i].shape[0] == 0:
        continue
    plt.plot(datas[i][:, 0], datas[i][:, 1], ".", color=colors[cmap_cnt], label=all_speaker[i])
    cmap_cnt += 1
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.legend()
plt.savefig(output_dir / f"data_colord_by_spk.{args.format}", bbox_inches='tight', pad_inches=0)
