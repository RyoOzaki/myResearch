import numpy as np
from scipy.io import wavfile
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()

parser.add_argument("--source_dir", type=Path, required=True)
parser.add_argument("--max_amplitude", type=float, default=30000)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--output_dir", type=Path)
group.add_argument("--replace", action="store_true")

parser.add_argument("--extension", type=str, default="wav")

args = parser.parse_args()

source_dir = args.source_dir

if args.replace:
    for file in source_dir.glob(f"**/*.{args.extension}"):
        rate, data = wavfile.read(file)
        max_amp = np.abs(data).max()
        data = data / max_amp * args.max_amplitude
        data = data.astype(np.int16)
        wavfile.write(file, rate, data)
else:
    for file in source_dir.glob(f"**/*.{args.extension}"):
        rate, data = wavfile.read(file)
        max_amp = np.abs(data).max()
        data = data / max_amp * args.max_amplitude
        data = data.astype(np.int16)

        relative_path = file.relative_to(source_dir)
        out_file = args.output_dir / relative_path
        out_file.parent.mkdir(exist_ok=True, parents=True)
        wavfile.write(out_file, rate, data)
