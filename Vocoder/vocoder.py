import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.io import wavfile
import pyworld as pw

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--samplerate", type=int, default=48000)
parser.add_argument("--frequency_dim", type=int, required=True)

parser.add_argument("--feature_file", type=Path, default="generated_feature.txt")

parser.add_argument("--frame_period", type=float, default=0.005)
# parser.add_argument("--normalize_max", type=float, default=200)
parser.add_argument("--ap_value", type=float, default=0.0)

parser.add_argument("--output", type=Path, required=True)

args = parser.parse_args()

mcep = np.loadtxt(args.feature_file)

mcep_f0 = mcep[:, 0].copy()
mcep_f0[mcep_f0 < 0] = 0

# mcep_f0 /= mcep_f0.max()
# mcep_f0 *= args.normalize_max

decoded_sp = pw.decode_spectral_envelope(mcep, args.samplerate, (args.frequency_dim - 1) * 2)
# synthesized_ap = np.full_like(decoded_sp, fill_value=args.ap_value) # if use argparse
synthesized_ap = np.full_like(decoded_sp, fill_value=0.5)
# synthesized_ap = np.random.rand(*decoded_sp.shape)
synthesized = pw.synthesize(mcep_f0, decoded_sp, synthesized_ap, args.samplerate, frame_period=args.frame_period*1000)

args.output.parent.mkdir(parents=True, exist_ok=True)

wavfile.write(args.output, args.samplerate, synthesized.astype(np.int16)*50)
