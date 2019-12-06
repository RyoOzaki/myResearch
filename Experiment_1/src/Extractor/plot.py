import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--target", type=Path, required=True)
parser.add_argument("--save_dir", type=Path, required=True)
parser.add_argument("--format", default="png")
parser.add_argument("--keep_dir", action="store_true")

args = parser.parse_args()

npz_obj = np.load(args.target)
save_dir = args.save_dir
save_dir.mkdir(parents=True, exist_ok=True)
format = args.format

for key, feat in npz_obj.items():
    print(key)

    plt.clf()
    plt.plot(feat)
    plt.title(key)

    if not args.keep_dir:
        key = key.replace("/", "_")
    outfile = save_dir / f"{key}.{format}"
    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(outfile)
