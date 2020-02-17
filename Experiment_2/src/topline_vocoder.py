import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.io import wavfile
import pyworld as pw
from scipy import signal
from util.utility import separate_speaker, get_separated_values
from Basic_generator.AP_generator import AP_generator
from Basic_generator.F0_generator import F0_generator
from Basic_generator.MCEP_generator import MCEP_generator

parser = ArgumentParser(fromfile_prefix_chars='@', formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--samplerate", type=int, default=48000)
parser.add_argument("--fftsize", type=int)
parser.add_argument("--frame_period", type=float, default=5E-3)

parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--ap", type=Path, required=True)
parser.add_argument("--f0", type=Path, required=True)
parser.add_argument("--mcep", type=Path, required=True)

parser.add_argument("--speaker_id", type=Path, required=True)

parser.add_argument("--target_speaker", type=str)

parser.add_argument("--output_prefix", type=Path)

parser.add_argument("--key_of_pickuped_sentences", nargs="+", required=True)

parser.add_argument("--mode", choices=["ML", "RND"], default="ML")

args = parser.parse_args()

speakers, spkind_keys = separate_speaker(np.load(args.speaker_id))
speaker_num = len(speakers)

param_flat = True
target_idx = speakers.index(args.target_speaker)
phn = np.load(args.phn)
phn_N = int(max(map(np.max, phn.values()))) + 1
gold_transcription = get_separated_values(phn, spkind_keys)[target_idx]
src_f0 = get_separated_values(np.load(args.f0), spkind_keys)[target_idx]
src_ap = get_separated_values(np.load(args.ap), spkind_keys)[target_idx]
src_mcep = get_separated_values(np.load(args.mcep), spkind_keys)[target_idx]

ap_generator = AP_generator(phn_N, src_ap, letter_stateseq=gold_transcription, flat=param_flat, mode=args.mode)
f0_generator = F0_generator(phn_N, src_f0, letter_stateseq=gold_transcription, flat=param_flat, mode=args.mode)
mcep_generator = MCEP_generator(phn_N, src_mcep, letter_stateseq=gold_transcription, gold_transcription=True, mode=args.mode)

sentences = args.key_of_pickuped_sentences
print(f"sentences: {sentences}")

for s, snt in enumerate(sentences):
    letter_stateseq = phn[snt].astype(int)

    mcep = mcep_generator.generate(letter_stateseq)
    ap = ap_generator.generate(letter_stateseq)
    f0 = f0_generator.generate(letter_stateseq)
    f0[f0<0] = 0

    mcep = signal.medfilt(mcep, (5, 1))
    mcep = mcep.astype(float, order="C")

    decoded_sp = pw.decode_spectral_envelope(mcep, args.samplerate, args.fftsize)
    synthesized = pw.synthesize(f0, decoded_sp, ap, args.samplerate, frame_period=args.frame_period*1000)
    synthesized = synthesized / max(abs(synthesized)) * 30000

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_file = args.output_prefix.with_name(f"{args.output_prefix.name}_{s:02d}_({snt.replace('/', '_')}).wav")

    wavfile.write(out_file, args.samplerate, synthesized.astype(np.int16))
