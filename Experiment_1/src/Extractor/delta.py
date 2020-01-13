import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from python_speech_features.base import delta

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--size", type=int, default=2)

args = parser.parse_args()

source = args.source
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)

source_npz = np.load(source)

output_dict = {}
for key, value in source_npz.items():
    output_dict[key] = delta(value, args.size)

np.savez(output, **output_dict)
