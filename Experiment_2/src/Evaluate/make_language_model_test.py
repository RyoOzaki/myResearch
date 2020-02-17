import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import shutil
import re
from itertools import product
import random

parser = ArgumentParser()

parser.add_argument("--source_dir", type=Path, required=True)
parser.add_argument("--speaker", type=str)
# parser.add_argument("--key_of_sentences", type=Path, required=True)

parser.add_argument("--cases", nargs="+", required=True)
parser.add_argument("--separate", type=int, default=1)
parser.add_argument("--num_of_test", type=int, required=True)

parser.add_argument("--output_dir", type=Path, required=True)

args = parser.parse_args()

root_dir = args.source_dir

# keys = args.key_of_sentences.read_text().split("\n")
# keys = [key for key in keys if key]

speakers = [args.speaker, ]

all_pairs = list(product(args.cases, speakers, range(args.num_of_test)))
random.shuffle(all_pairs)

for s in range(args.separate):
    separated_pairs = all_pairs[s::args.separate]
    if args.separate == 1:
        output_dir = args.output_dir
    else:
        output_dir = args.output_dir.with_name(f"{args.output_dir.name}_{s+1}")
    output_dir.mkdir(exist_ok=True, parents=True)
    for i, (case, spk, idx) in enumerate(separated_pairs):
        source_dir = args.source_dir / case / spk
        f = list(source_dir.glob(f"out_{idx:02d}_*.wav"))[0]
        out_file = output_dir / f"test_{i:02d}.wav"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(str(f), str(out_file))

    test_datas = ["File_name, MOS, Neary"]
    test_format = "{file_name}, {MOS}, {neary}"

    datas = ["Case, Target_speaker, MOS, Neary"]
    format = "{case}, {target_speaker}, {MOS}, {neary}"
    for i, (case, spk, idx) in enumerate(separated_pairs):
        datas.append(format.format(case=case, target_speaker=spk, MOS="", neary=""))

        test_datas.append(test_format.format(file_name=f"test_{i:02d}", MOS="", neary=""))

    out_file = output_dir / "No_touch_me.csv"
    out_file.write_text("\n".join(datas))

    out_file = output_dir / "mos_neary.csv"
    out_file.write_text("\n".join(test_datas))
