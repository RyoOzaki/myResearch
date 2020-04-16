import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from modules.loader import *
from modules.utility import *
import re
from math import ceil, floor
import pyworld as pw

def _load_label_list(files, sp=None):
    s = set()
    for f in files:
        raw_label = _load_raw_label(f)
        s |= set([raw[2] for raw in raw_label])
    l = list(s)
    l.sort()
    if sp is not None and type(sp) is dict:
        s -= set(sp.values())
        l = list(s)
        l.sort()
        for k in sorted(sp.keys()):
            l.insert(k, sp[k])
    return l

def _load_raw_label(f):
    label_pattern = r"^(?P<begin_frame>[\d\.]+)\s+(?P<end_frame>[\d\.]+)\s(?P<label>.*)$"
    body = f.read_text()
    lines = body.split("\n")
    lst = []
    for line in lines:
        if line.startswith("#"):
            continue
        m = re.search(label_pattern, line)
        if m:
            bf = eval(m.group("begin_frame"))
            ef = eval(m.group("end_frame"))
            lab = m.group("label")
            lst.append((bf, ef, lab))
    return lst

def _label_cord(label_file, label_list, length, window_frame, step_frame, init_val=0):
    label_ary = np.ones(length, dtype=int) * init_val
    Ft = np.zeros(length, dtype=int)
    raw_labels = _load_raw_label(label_file)
    for b,e,l in raw_labels:
        left = int(max(0, floor((2*b-window_frame)/(2*step_frame))))
        right = int(ceil((2*e-window_frame)/(2*step_frame)))
        label_ary[left:right] = label_list.index(l)
        if right-1 < length:
            Ft[right-1] = 1
    return label_ary, Ft

def _label_cord_mfcc_frame(label_file, label_list, length, init_val=0):
    label_ary = np.ones(length, dtype=int) * init_val
    Ft = np.zeros(length, dtype=int)
    raw_labels = _load_raw_label(label_file)
    for b,e,l in raw_labels:
        label_ary[b:e+1] = label_list.index(l)
        if e < length:
            Ft[e] = 1
    return label_ary, Ft

default_parameters = {
    "samplerate": None,
    "frame_period": 0.005,
    "nfilt": 26,
    "fftsize": 1024,
}

parameters_types = {
    "samplerate": int,
    "frame_period": float,
    "nfilt": int,
    "fftsize": int
}

parameters_help = {
    "samplerate": "the sample rate of the signal in extracting. if it is lower than the wave files one, it do downsampling.",
    "frame_period": "the length of the analysis window in seconds.",
    "nfilt": "the number of filters in the filterbank.",
    "fftsize": "the number of fft window size"
}

enabled_feature_types = ["mcep", "spenv", "f0", "ap"]

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source_dir", type=Path, default=Path("./"), help="source directory of wave files. Default is './'")

parser.add_argument("--format", default="wave", choices=["wave", "sph"], help="format of input files. Default is wave. you can choice [wave/sph]")
parser.add_argument("--extension", default="wav", help="extension of input files")

parser.add_argument("--feature_type", nargs="+", default=["mcep",], choices=["all", *enabled_feature_types], help="output format of speech features.")

parser.add_argument("--label_format", default="none", choices=["none", "time", "wave_frame", "mfcc_frame"], help="format of label")
parser.add_argument("--phn_label_extension", help="extension of phoneme label")
parser.add_argument("--wrd_label_extension", help="extension of word label")
parser.add_argument("--sil_label", help="silent label name (or value) in label files")

for name in default_parameters:
    value = default_parameters[name]
    help = parameters_help[name]
    dtype = parameters_types[name]
    parser.add_argument(
        f"--{name}",
        type=dtype,
        default=value,
        help=help
    )

args = parser.parse_args()

source_dir = args.source_dir
format = args.format
extension = args.extension
feature_type = args.feature_type
if "all" in feature_type:
    feature_type = enabled_feature_types

if args.format == "wave":
    load_function = load_wav
elif args.format == "sph":
    load_function = load_sph

vargs = vars(args)
parameters = {}
for name in default_parameters:
    parameters[name] = vargs[name]

if args.label_format != "none":
    if args.sil_label:
        sp_kwargs = {0: args.sil_label}
    else:
        sp_kwargs = {}
    if args.phn_label_extension is None or args.wrd_label_extension is None:
        raise ValueError("If you specified --label_format, you must specify --phn_label_extension and --wrd_label_extension.")
    phn_label_dict = _load_label_list(source_dir.glob(f"**/*.{args.phn_label_extension}"), sp=sp_kwargs)
    wrd_label_dict = _load_label_list(source_dir.glob(f"**/*.{args.wrd_label_extension}"), sp=sp_kwargs)

    print(f"phn_label_list : {phn_label_dict}")
    print(f"wrd_label_list : {wrd_label_dict}")

cnt = 0
print(f"extracting features: {feature_type}")
for file in source_dir.glob(f"**/*.{extension}"):
    print(f"{file}")
    cnt += 1
    N = None
    fs, signal = load_function(str(file))
    signal = signal.astype(np.float)
    samplerate = args.samplerate or fs # if the samplerate is specified, use specified one, and else, use the wave files one.
    signal = downsampling(signal, fs, samplerate) # fs -> samplerate
    signal = convert2mono(signal)

    _f0, t = pw.dio(signal, samplerate, frame_period=args.frame_period*1000) # 基本周波数の抽出
    # _f0, t = pw.harvest(signal, samplerate, frame_period=args.frame_period*1000) # 基本周波数の抽出
    f0 = pw.stonemask(signal, _f0, t, samplerate) # 基本周波数の修正
    sp = pw.cheaptrick(signal, f0, t, samplerate, fft_size=args.fftsize)  # スペクトル包絡spectrumの抽出
    ap = pw.d4c(signal, f0, t, samplerate, fft_size=args.fftsize)  # 非周期性指標の抽出
    mcep = pw.code_spectral_envelope(sp, samplerate, args.nfilt) # メルケプストラムの抽出

    N = mcep.shape[0]

    if "spenv" in feature_type:
        np.savetxt(file.with_suffix(".spenv"), sp)

    if "mcep" in feature_type:
        np.savetxt(file.with_suffix(".mcep"), mcep)

    if "f0" in feature_type:
        np.savetxt(file.with_suffix(".f0"), f0)

    if "ap" in feature_type:
        np.savetxt(file.with_suffix(".ap"), ap)

    if args.label_format != "none":
        if args.label_format == "time":
            frame_len = args.frame_period
        elif args.label_format == "wave_frame":
            frame_len = int(args.frame_period * fs)

        if args.phn_label_extension:
            phn_file = file.with_suffix(f".{args.phn_label_extension}")
            if args.label_format == "mfcc_frame":
                phn, Ft = _label_cord_mfcc_frame(phn_file, phn_label_dict, N)
            else:
                phn, Ft = _label_cord(phn_file, phn_label_dict, N, frame_len, frame_len)
            np.savetxt(file.with_suffix(".phn"), phn, fmt="%d")
            np.savetxt(file.with_suffix(".Ft_phn"), Ft, fmt="%d")

        if args.wrd_label_extension:
            wrd_file = file.with_suffix(f".{args.wrd_label_extension}")
            if args.label_format == "mfcc_frame":
                wrd, Ft = _label_cord_mfcc_frame(wrd_file, wrd_label_dict, N)
            else:
                wrd, Ft = _label_cord(wrd_file, wrd_label_dict, N, frame_len, frame_len)
            np.savetxt(file.with_suffix(".wrd"), wrd, fmt="%d")
            np.savetxt(file.with_suffix(".Ft_wrd"), Ft, fmt="%d")

print(f"{cnt} files were process.")
