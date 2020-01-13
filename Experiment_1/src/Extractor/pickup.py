import numpy as np
from pathlib import Path
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def regexp_match_checker(keyword, key):
    return re.match(keyword, key)

def in_checker(keyword, key):
    return (keyword in key)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--regexp", action="store_true")
parser.add_argument("--keyword", required=True)

args = parser.parse_args()

source = args.source
output = args.output
keyword = args.keyword

output.parent.mkdir(parents=True, exist_ok=True)

source_npz = np.load(source)

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

np.savez(output, **output_dict)
print(f"{i} files are pickuped!!")
