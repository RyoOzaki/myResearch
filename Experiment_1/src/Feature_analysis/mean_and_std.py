import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from util.utility import separate_speaker, get_separated_values
import itertools

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--phn", type=Path)
parser.add_argument("--speaker_id", type=Path)
parser.add_argument("--mode", type=str, choices=["none", "phn", "spk"])

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

dim = source[keys[0]].shape[1]
if mode == "none":
    datas = np.concatenate([source[key] for key in keys], axis=0)
    print(datas.mean(axis=0))
    print(np.cov(datas, rowvar=False, bias=True))
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

    for i in range(N):
        print(f"======={i+1}======")
        if datas[i].shape[0] == 0:
            continue
        print(datas[i].mean(axis=0))
        print(np.cov(datas[i], rowvar=False, bias=True))
