import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.io import wavfile

plt.rcParams["font.size"] = 18

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)

args = parser.parse_args()

fs, source_wav = wavfile.read(args.source)
if source_wav.ndim == 2:
    source_wav = source_wav.mean(axis=1)
length = source_wav.shape[0]
x = np.arange(length)
x_ticks_span = 500 # msec
x_ticks_span_f = int(fs * x_ticks_span * 1E-3)
x_ticks_max = np.ceil(length / x_ticks_span_f)
x_ticks = np.arange(int(x_ticks_max)) * x_ticks_span # msec
x_ticks = [f"{v*1E-3:.1f}" for v in x_ticks]
plt.plot(x, source_wav, "-")
plt.xticks(x[::x_ticks_span_f], x_ticks)
plt.ylim(-30000, 30000)
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.gca().ticklabel_format(style="sci", scilimits=(0,0), axis="y")
plt.tight_layout()

args.output.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(args.output)
