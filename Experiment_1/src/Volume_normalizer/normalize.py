import numpy as np
from argparse import ArgumentParser
from pathlib import Path

def load_wav(f):
    import scipy.io.wavfile as wav
    (rate,data) = wav.read(f)
    return (rate, data)

def load_sph(f):
    from sphfile import SPHFile
    sph = SPHFile(f)
    return (sph.format['sample_rate'], sph.content)

parser = ArgumentParser()

parser.add_argument("--source_dir", type=Path, required=True)
parser.add_argument("--max_amplitude", type=float, default=30000)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--output_dir", type=Path)
group.add_argument("--replace", action="store_true")

parser.add_argument("--extension", type=str, default="wav")

parser.add_argument("--format", default="wave", choices=["wave", "sph"], help="format of input files. Default is wave. you can choice [wave/sph]")

args = parser.parse_args()

if args.format == "wave":
    load_function = load_wav
elif args.format == "sph":
    load_function = load_sph


source_dir = args.source_dir

if args.replace:
    for file in source_dir.glob(f"**/*.{args.extension}"):
        rate, data = load_function(str(file))
        max_amp = np.abs(data).max()
        data = data / max_amp * args.max_amplitude
        data = data.astype(np.int16)
        wavfile.write(file, rate, data)
else:
    for file in source_dir.glob(f"**/*.{args.extension}"):
        rate, data = load_function(str(file))
        max_amp = np.abs(data).max()
        data = data / max_amp * args.max_amplitude
        data = data.astype(np.int16)

        relative_path = file.relative_to(source_dir)
        out_file = args.output_dir / relative_path
        out_file.parent.mkdir(exist_ok=True, parents=True)
        wavfile.write(out_file, rate, data)
