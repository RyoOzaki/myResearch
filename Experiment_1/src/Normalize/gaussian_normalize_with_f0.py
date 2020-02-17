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

def normalize(source_datas, source_f0):
    packed_datas, lengths = packing(source_datas)
    packed_f0, f0_lengths = packing(source_f0)
    assert lengths == f0_lengths

    mean = packed_datas[packed_f0 != 0].mean(axis=0)
    std = packed_datas[packed_f0 != 0].std(axis=0)
    normalized_datas = (packed_datas - mean) / std

    unpacked_normalized_datas = unpacking(normalized_datas, lengths)

    lf0 = np.log(packed_f0[packed_f0 != 0])

    return unpacked_normalized_datas, mean, std, lf0.mean(), lf0.std()

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--source_f0", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)

parser.add_argument("--output", type=Path)

args = parser.parse_args()

if args.output is None:
    if args.speaker_id is None:
        out_file = args.source.with_name(f"gaunorm_with_f0_{args.source.stem}.npz")
    else:
        out_file = args.source.with_name(f"gaunorm_with_f0_spkind_{args.source.stem}.npz")
else:
    out_file = args.output
out_file.parent.mkdir(exist_ok=True, parents=True)
param_dir = out_file.with_suffix("")
param_dir.mkdir(exist_ok=True)

npz_obj = np.load(args.source)
f0_npz_obj = np.load(args.source_f0)
keys = sorted(list(npz_obj.keys()))

normalized = {}
if args.speaker_id is not None:
    speaker_npz = np.load(args.speaker_id)
    unique_speakers = set(list(map(str, speaker_npz.values())))

    parameter_means = {}
    parameter_stds = {}
    lf0_means = {}
    lf0_stds = {}
    for spk in unique_speakers:
        speaker_individual_keys = [key for key in keys if speaker_npz[key] == spk]
        norm_datas, mean, std, lf0_mean, lf0_std = normalize([npz_obj[key] for key in speaker_individual_keys], [f0_npz_obj[key] for key in speaker_individual_keys])
        for key, data in zip(speaker_individual_keys, norm_datas):
            normalized[key] = data
        parameter_means[spk] = mean
        parameter_stds[spk] = std
        lf0_means[spk] = lf0_mean
        lf0_stds[spk] = lf0_std

    np.savez(out_file, **normalized)
    np.savez(param_dir / "means.npz", **parameter_means)
    np.savez(param_dir / "stds.npz", **parameter_stds)
    np.savez(param_dir / "lf0_means.npz", **lf0_means)
    np.savez(param_dir / "lf0_stds.npz", **lf0_stds)
else:
    norm_datas, mean, std, lf0_mean, lf0_std = normalize([npz_obj[key] for key in keys], [f0_npz_obj[key] for key in keys])
    for key, data in zip(keys, norm_datas):
        normalized[key] = data

    np.savez(out_file, **normalized)
    np.save(param_dir / "mean.npy", mean)
    np.save(param_dir / "std.npy", std)
    np.save(param_dir / "lf0_mean.npy", lf0_mean)
    np.save(param_dir / "lf0_std.npy", lf0_std)
