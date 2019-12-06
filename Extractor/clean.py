from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--source_dir", type=Path, default=Path("./"), help="source directory of wave files. Default is './'")
parser.add_argument("--remove_extensions", nargs="*", default=["mfcc", "dmfcc", "ddmfcc", "mcep", "mspec", "logmspec", "spenv", "ap", "f0", "phn", "wrd", "npz"], help="target extensions")

args = parser.parse_args()

cnt = 0
for ext in args.remove_extensions:
    for file in args.source_dir.glob(f"**/*.{ext}"):
        if file.is_file():
            print(file)
            cnt += 1
            file.unlink()
print(f"{cnt} files were delete.")
