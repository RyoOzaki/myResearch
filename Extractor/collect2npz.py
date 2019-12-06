import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source_dir", type=Path, default=Path("./"), help="source directory of wave files. Default is './'")
parser.add_argument("--output_dir", type=Path, help="output directory of npz file. Default is same as source_dir")
parser.add_argument("--collect_extensions", nargs="+", default=["mfcc", "dmfcc", "ddmfcc", "mcep", "mspec", "logmspec", "spenv", "f0", "ap", "phn", "wrd"], help="collect extensions")
parser.add_argument("--with_speaker_id", action="store_true", help="if you specified this option, save the speaker_id npz object")
parser.add_argument("--speaker_dir_layer", type=int, default=1, help="layer of speaker. e.g., source/speaker_1, and you specified source_dir to source/, speaker_dir_layer is 1.")

args = parser.parse_args()

source_dir = args.source_dir
output_dir = args.output_dir or source_dir
output_dir.mkdir(parents=True, exist_ok=True)
with_speaker_id = args.with_speaker_id
speaker_dir_layer = args.speaker_dir_layer
collect_extensions = args.collect_extensions

if with_speaker_id:
    speakers = {}

for ext in collect_extensions:
    datas = {}
    cnt = 0
    for file in source_dir.glob(f"**/*.{ext}"):
        cnt += 1
        relative_path = file.relative_to(source_dir)
        datas_key = str(relative_path.with_suffix(""))
        if with_speaker_id:
            speaker = relative_path.parts[speaker_dir_layer-1]
            speakers[datas_key] = speaker
        datas[datas_key] = np.loadtxt(file)
    if len(datas) != 0:
        np.savez(output_dir / f"{ext}.npz", **datas)
        print(f"{ext} files: {cnt} files were collect.")

if with_speaker_id:
    np.savez(output_dir / "speaker_id.npz", **speakers)
