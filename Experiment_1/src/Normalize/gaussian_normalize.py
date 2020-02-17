import numpy as np
from pathlib import Path
from argparse import ArgumentParser

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

def normalize(source_datas):
    packed_datas, lengths = packing(source_datas)

    mean = packed_datas.mean(axis=0)
    std = packed_datas.std(axis=0)
    normalized_datas = (packed_datas - mean) / std

    unpacked_normalized_datas = unpacking(normalized_datas, lengths)

    return unpacked_normalized_datas, mean, std

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)

parser.add_argument("--output", type=Path)

args = parser.parse_args()

if args.output is None:
    if args.speaker_id is None:
        out_file = args.source.with_name(f"gaunorm_{args.source.stem}.npz")
    else:
        out_file = args.source.with_name(f"gaunorm_spkind_{args.source.stem}.npz")
else:
    out_file = args.output
out_file.parent.mkdir(exist_ok=True, parents=True)
param_dir = out_file.with_suffix("")
param_dir.mkdir(exist_ok=True)

npz_obj = np.load(args.source)
keys = sorted(list(npz_obj.keys()))

normalized = {}
if args.speaker_id is not None:
    speaker_npz = np.load(args.speaker_id)
    unique_speakers = set(list(map(str, speaker_npz.values())))

    parameter_means = {}
    parameter_stds = {}
    for spk in unique_speakers:
        speaker_individual_keys = [key for key in keys if speaker_npz[key] == spk]
        norm_datas, mean, std = normalize([npz_obj[key] for key in speaker_individual_keys])
        for key, data in zip(speaker_individual_keys, norm_datas):
            normalized[key] = data
        parameter_means[spk] = mean
        parameter_stds[spk] = std

    np.savez(out_file, **normalized)
    np.savez(param_dir / "means.npz", **parameter_means)
    np.savez(param_dir / "stds.npz", **parameter_stds)
else:
    norm_datas, mean, std = normalize([npz_obj[key] for key in keys])
    for key, data in zip(keys, norm_datas):
        normalized[key] = data

    np.savez(out_file, **normalized)
    np.save(param_dir / "mean.npy", mean)
    np.save(param_dir / "std.npy", std)
