import numpy as np
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--sources", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)

args = parser.parse_args()

sources_npz = [np.load(source) for source in args.sources]
keys = sorted(list(sources_npz[0].keys()))

concatenated = {}

for key in keys:
    concatenated[key] = np.concatenate([source[key] for source in sources_npz], axis=1)

np.savez(args.output, **concatenated)
