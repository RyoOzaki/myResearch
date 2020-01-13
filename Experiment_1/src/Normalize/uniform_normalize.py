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

def normalize(source_datas, min_value, max_value):
    packed_datas, lengths = packing(source_datas)

    min = packed_datas.min(axis=0)
    max = packed_datas.max(axis=0)
    normalized_datas = (packed_datas - min) / (max - min)
    normalized_datas = (max_value - min_value) * normalized_datas + min_value

    unpacked_normalized_datas = unpacking(normalized_datas, lengths)

    return unpacked_normalized_datas, min, max

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)

parser.add_argument("--min_value", type=float, default=0.0)
parser.add_argument("--max_value", type=float, default=1.0)

parser.add_argument("--output", type=Path)

args = parser.parse_args()

out_file = args.output or args.source.with_name(f"uninormalized_{args.source.stem}.npz")
out_file.parent.mkdir(exist_ok=True, parents=True)
param_dir = out_file.with_suffix("")
param_dir.mkdir(exist_ok=True)

npz_obj = np.load(args.source)
keys = sorted(list(npz_obj.keys()))

normalized = {}
if args.speaker_id is not None:
    speaker_npz = np.load(args.speaker_id)
    unique_speakers = set(list(map(str, speaker_npz.values())))

    parameter_mins = {}
    parameter_maxs = {}
    for spk in unique_speakers:
        speaker_individual_keys = [key for key in keys if speaker_npz[key] == spk]
        norm_datas, min, max = normalize([npz_obj[key] for key in speaker_individual_keys], args.min_value, args.max_value)
        for key, data in zip(speaker_individual_keys, norm_datas):
            normalized[key] = data
        parameter_mins[spk] = min
        parameter_maxs[spk] = max

    np.savez(out_file, **normalized)
    np.savez(param_dir / "mins.npz", **parameter_mins)
    np.savez(param_dir / "maxs.npz", **parameter_maxs)
else:
    norm_datas, min, max = normalize([npz_obj[key] for key in keys], args.min_value, args.max_value)
    for key, data in zip(keys, norm_datas):
        normalized[key] = data

    np.savez(out_file, **normalized)
    np.save(param_dir / "min.npy", min)
    np.save(param_dir / "max.npy", max)
