import numpy as np
from pathlib import Path
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def regexp_match_checker(keyword, key):
    return re.match(keyword, key)

def in_checker(keyword, key):
    return (keyword in key)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source_file", type=Path, required=True)
parser.add_argument("--output_file", type=Path, required=True)
parser.add_argument("--regexp", action="store_true")
parser.add_argument("--keyword", required=True)

args = parser.parse_args()

source_file = args.source_file
output_file = args.output_file
keyword = args.keyword

output_file.parent.mkdir(parents=True, exist_ok=True)

source_npz = np.load(source_file)

if args.regexp:
    checker = regexp_match_checker
else:
    checker = in_checker

output_dict = {}
i = 0
for key, value in source_npz.items():
    if checker(keyword, key):
        output_dict[key] = value
        i += 1

np.savez(output_file, **output_dict)
print(f"{i} files are pickuped!!")
