import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import re

scores = ["letter_ARI", "letter_micro_F1", "letter_macro_F1", "word_ARI", "word_micro_F1", "word_macro_F1", "resample_times", "log_likelihoods"]

parser = ArgumentParser()

parser.add_argument("--result_dir", type=Path, required=True)
parser.add_argument("--iteration", type=int, default=-1)
parser.add_argument("--score", nargs="+", default=["letter_ARI", "word_ARI"], choices=scores)
parser.add_argument("--float_format", default=".3f")
parser.add_argument("--pm_str", default="±")

parser.add_argument("--prefix", default="")
parser.add_argument("--suffix", default="")

parser.add_argument("--keyword", default="")
parser.add_argument("--exclude_keyword")

parser.add_argument("--with_max", action="store_true")
parser.add_argument("--with_min", action="store_true")
parser.add_argument("--with_map", action="store_true")
parser.add_argument("--with_corr", action="store_true")

args = parser.parse_args()

result_dir = args.result_dir
format_str     = f"  {{name}}: {args.prefix}{{mean:{args.float_format}}} {args.pm_str} {{std:{args.float_format}}}{args.suffix}"
sub_format_str = f"    {{label}}: {{value:{args.float_format}}} (trial idx: {{trial_idx}})"
format_corr    = f"    corr: {{corr:{args.float_format}}}"

for score in args.score:
    print(f"{score}")
    npz_obj = np.load(result_dir / f"{score}.npz")
    ll_npz_obj = np.load(result_dir / "log_likelihoods.npz")
    keys = sorted(list(npz_obj.keys()))
    for key in keys:
        if args.exclude_keyword is not None and re.match(f".*{args.exclude_keyword}.*", key):
            continue
        if re.match(f".*{args.keyword}.*", key):
            target_data = npz_obj[key][:, args.iteration]
            target_ll = ll_npz_obj[key][:, args.iteration]
            print(format_str.format(name=key, mean=target_data.mean(), std=target_data.std()))
            if args.with_corr:
                print(format_corr.format(corr=np.corrcoef(target_data, target_ll)[0, 1]))
            if args.with_max:
                print(sub_format_str.format(label="max", value=target_data.max(), trial_idx=target_data.argmax()+1))
                # print(f"    max: {target_data.max():args.float_format} (trial idx: {target_data.argmax()+1})")
            if args.with_min:
                print(sub_format_str.format(label="min", value=target_data.min(), trial_idx=target_data.argmin()+1))
                # print(f"    min: {target_data.min():args.float_format} (trial idx: {target_data.argmin()+1})")
            if args.with_map:
                map_idx = target_ll.argmax()
                print(sub_format_str.format(label="map", value=target_data[map_idx], trial_idx=map_idx+1))
                # print(f"    map: {target_data[map_idx]:args.float_format} (trial idx: {map_idx+1})")
    print()
