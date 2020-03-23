import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from python_speech_features import mfcc as psf_mfcc
from python_speech_features import fbank as psf_fbank
from python_speech_features import logfbank as psf_logfbank
from python_speech_features.base import delta as psf_delta
from modules.loader import *
from modules.utility import *
import re
from math import ceil, floor

def padding(signal, samplerate, winlen, winstep):
    flen = int(samplerate * winlen)
    fstep = int(samplerate * winstep)
    mod = (signal.shape[0] - flen) % fstep
    pad = np.zeros((fstep - mod, ), dtype=signal.dtype)
    signal = np.concatenate([signal, pad], axis=0)
    return signal

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
        if right > length:
            right = length
        Ft[right-1] = 1
    return label_ary, Ft

def _label_cord_mfcc_frame(label_file, label_list, length, init_val=0):
    label_ary = np.ones(length, dtype=int) * init_val
    Ft = np.zeros(length, dtype=int)
    raw_labels = _load_raw_label(label_file)
    for b,e,l in raw_labels:
        label_ary[b:e+1] = label_list.index(l)
        Ft[e] = 1
    return label_ary, Ft

default_parameters = {
    "samplerate": None,
    "winlen": 0.025,
    "winstep": 0.01,
    "numcep": 13,
    "nfilt": 26,
    "nfft": None,
    "lowfreq": 0.0,
    "highfreq": None,
    "preemph": 0.97,
    "ceplifter": 22,
    "appendEnergy": False,
    "winfunc": "hamming"
}

parameters_types = {
    "samplerate": int,
    "winlen": float,
    "winstep": float,
    "numcep": int,
    "nfilt": int,
    "nfft": int,
    "lowfreq": float,
    "highfreq": float,
    "preemph": float,
    "ceplifter": int,
    "appendEnergy": bool,
    "winfunc": str
}

parameters_help = {
    "samplerate": "the sample rate of the signal in extracting. if it is lower than the wave files one, it do downsampling.",
    "winlen": "the length of the analysis window in seconds.",
    "winstep": "the step between successive windows in seconds.",
    "numcep": "the number of cepstrum to return.",
    "nfilt": "the number of filters in the filterbank.",
    "nfft": "the FFT size. Default is same as winlen.",
    "lowfreq": "lowest band edge of mel filters. In Hz.",
    "highfreq": "highest band edge of mel filters. In Hz, default is samplerate/2",
    "preemph": "apply preemphasis filter with preemph as coefficient. 0 is no filter.",
    "ceplifter": "apply a lifter to final cepstral coefficients. 0 is no lifter.",
    "appendEnergy": "if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy",
    "winfunc": "the analysis window to apply to each frame. You can use [hamming/none]"
}

candidates = {
    "winfunc": ["hamming", "none"]
}

enabled_feature_types = ["mfcc", "mspec", "logmspec"]

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source_dir", type=Path, default=Path("./"), help="source directory of wave files. Default is './'")

parser.add_argument("--format", default="wave", choices=["wave", "sph"], help="format of input files. Default is wave. you can choice [wave/sph]")
parser.add_argument("--extension", default="wav", help="extension of input files")

parser.add_argument("--feature_type", nargs="+", default=["mfcc",], choices=["all", *enabled_feature_types], help="output format of speech features.")

parser.add_argument("--label_format", default="none", choices=["none", "time", "wave_frame", "mfcc_frame"], help="format of label")
parser.add_argument("--phn_label_extension", help="extension of phoneme label")
parser.add_argument("--wrd_label_extension", help="extension of word label")
parser.add_argument("--sil_label", help="silent label name (or value) in label files")

for name in default_parameters:
    value = default_parameters[name]
    help = parameters_help[name]
    dtype = parameters_types[name]
    if name in candidates:
        parser.add_argument(
            f"--{name}",
            type=dtype,
            default=value,
            choices=candidates[name],
            help=help,
        )
    else:
        parser.add_argument(
            f"--{name}",
            type=dtype,
            default=value,
            help=help
        )
parser.add_argument("--delta_winlen", type=int, default=2, help="delta_window size")

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

if parameters["winfunc"] == "hamming":
    parameters["winfunc"] = np.hamming
elif parameters["winfunc"] == "none":
    parameters["winfunc"] = lambda x: np.ones((x,))

if args.label_format != "none":
    if args.sil_label:
        sp_kwargs = {0: args.sil_label}
    else:
        sp_kwargs = {}
    if args.phn_label_extension is None or args.wrd_label_extension is None:
        raise ValueError("If you specified --label_format, you must specify --phn_label_extension and --wrd_label_extension.")
    phn_label_dict = _load_label_list(source_dir.glob(f"**/*.{args.phn_label_extension}"), sp=sp_kwargs)
    wrd_label_dict = _load_label_list(source_dir.glob(f"**/*.{args.wrd_label_extension}"), sp=sp_kwargs)

cnt = 0
print(f"extracting features: {feature_type}")
for file in source_dir.glob(f"**/*.{extension}"):
    print(f"{file}")
    cnt += 1
    N = None
    fs, signal = load_function(str(file))
    samplerate = args.samplerate or fs # if the samplerate is specified, use specified one, and else, use the wave files one.
    signal = downsampling(signal, fs, samplerate) # fs -> samplerate
    signal = convert2mono(signal)
    signal = padding(signal, samplerate, parameters["winlen"], parameters["winstep"])
    nfft = parameters["nfft"] or int(samplerate * parameters["winlen"])

    if "mfcc" in feature_type:
        mfcc = psf_mfcc(signal, **{**parameters, "samplerate": samplerate, "nfft": nfft})
        delta_window_size = args.delta_winlen
        dmfcc = psf_delta(mfcc, delta_window_size)
        ddmfcc = psf_delta(dmfcc, delta_window_size)
        N = N or mfcc.shape[0]
        np.savetxt(file.with_suffix(".mfcc"), mfcc)
        np.savetxt(file.with_suffix(".dmfcc"), dmfcc)
        np.savetxt(file.with_suffix(".ddmfcc"), ddmfcc)

    # for mspec and logmspec
    need_parameters = ["samplerate", "winlen", "winstep", "nfilt", "nfft", "lowfreq", "highfreq", "preemph", "winfunc"]
    sub_parametes = {}
    for name in need_parameters:
        sub_parametes[name] = parameters[name]

    if "mspec" in feature_type:
        mspec, energy = psf_fbank(signal, **{**sub_parametes, "samplerate": samplerate, "nfft": nfft})
        N = N or mspec.shape[0]
        np.savetxt(file.with_suffix(".mspec"), mspec)

    if "logmspec" in feature_type:
        logmspec = psf_logfbank(signal, **{**sub_parametes, "samplerate": samplerate, "nfft": nfft})
        N = N or logmspec.shape[0]
        np.savetxt(file.with_suffix(".logmspec"), logmspec)

    if args.label_format != "none":
        if args.label_format == "time":
            window_len = parameters["winlen"]
            step_len = parameters["winstep"]
        elif args.label_format == "wave_frame":
            window_len = int(parameters["winlen"] * fs)
            step_len = int(parameters["winstep"] * fs)

        if args.phn_label_extension:
            phn_file = file.with_suffix(f".{args.phn_label_extension}")
            if args.label_format == "mfcc_frame":
                phn, Ft = _label_cord_mfcc_frame(phn_file, phn_label_dict, N)
            else:
                phn, Ft = _label_cord(phn_file, phn_label_dict, N, window_len, step_len)
            np.savetxt(file.with_suffix(".phn"), phn, fmt="%d")
            np.savetxt(file.with_suffix(".Ft_phn"), Ft, fmt="%d")

        if args.wrd_label_extension:
            wrd_file = file.with_suffix(f".{args.wrd_label_extension}")
            if args.label_format == "mfcc_frame":
                wrd, Ft = _label_cord_mfcc_frame(wrd_file, wrd_label_dict, N)
            else:
                wrd, Ft = _label_cord(wrd_file, wrd_label_dict, N, window_len, step_len)
            np.savetxt(file.with_suffix(".wrd"), wrd, fmt="%d")
            np.savetxt(file.with_suffix(".Ft_wrd"), Ft, fmt="%d")

print(f"{cnt} files were process.")
