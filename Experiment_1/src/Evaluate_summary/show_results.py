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

parser.add_argument("--keyword", default="")

args = parser.parse_args()

result_dir = args.result_dir
format_str = f"  {{name}}: {{mean:{args.float_format}}} ± {{std:{args.float_format}}}"

for score in args.score:
    print(f"{score}")
    npz_obj = np.load(result_dir / f"{score}.npz")
    for key in npz_obj:
        if re.match(f".*{args.keyword}.*", key):
            target_data = npz_obj[key][:, args.iteration]
            print(format_str.format(name=key, mean=target_data.mean(), std=target_data.std()))
    print()
