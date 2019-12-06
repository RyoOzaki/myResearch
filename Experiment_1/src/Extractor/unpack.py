import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output_dir", type=Path)
parser.add_argument("--keep_dir", action="store_true")

args = parser.parse_args()

source_file = args.source
output_dir = args.output_dir or source_file.parent

stem = source_file.stem

n = 0
source_npz = np.load(source_file)
for path, value in source_npz.items():
    n += 1
    if not args.keep_dir:
        path = path.replace("/", "_")
    output_path = (output_dir / path).with_suffix(f".{stem}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(output_path)
    np.savetxt(output_path, value)
print(f"unpacked {n} files.")
