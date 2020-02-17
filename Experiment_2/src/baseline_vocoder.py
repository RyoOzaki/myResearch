import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from scipy.io import wavfile
import pyworld as pw
from scipy import signal
from util.utility import separate_speaker, get_separated_values
from LSTMLM.LSTMLM_generater import LSTMLM_generator
from Basic_generator.AP_generator import AP_generator
from Basic_generator.F0_generator import F0_generator
from Basic_generator.MCEP_generator import MCEP_generator
from NPBDAA_generator.NPBDAA_generator import NPBDAA_generator
from NPBDAA_LM.Bigram_generator import Bigram_generator
from NPBDAA_LM.Unigram_generator import Unigram_generator

def expanding_label(label_src, width=4):
    ones = np.ones((label_src.shape[0], width), dtype=int)
    labs = ones * label_src.reshape((-1, 1))
    return labs.reshape((-1, ))

parser = ArgumentParser(fromfile_prefix_chars='@', formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--samplerate", type=int, default=48000)
parser.add_argument("--fftsize", type=int)
parser.add_argument("--frame_period", type=float, default=5E-3)

parser.add_argument("--sentences_file", type=Path, required=True)

parser.add_argument("--letter_num", type=int, required=True)
parser.add_argument("--letter_stateseq", type=Path, required=True)
parser.add_argument("--ap", type=Path, required=True)
parser.add_argument("--f0", type=Path, required=True)
parser.add_argument("--mcep", type=Path, required=True)

parser.add_argument("--parameter", type=Path)

parser.add_argument("--speaker_id", type=Path, required=True)

parser.add_argument("--target_speaker", type=str)

parser.add_argument("--output_prefix", type=Path)

parser.add_argument("--sentences", action="append", type=int, nargs="+")
parser.add_argument("--size", type=int, default=1)

parser.add_argument("--mode", choices=["ML", "RND"], default="ML")

parser.add_argument("--LM", choices=["LSTM", "Bigram", "Unigram"])

parser.add_argument("--LSTM_model", type=Path)

args = parser.parse_args()

speakers, spkind_keys = separate_speaker(np.load(args.speaker_id))
speaker_num = len(speakers)

param_flat = True
target_idx = speakers.index(args.target_speaker)
src_letter_stateseq = get_separated_values(np.load(args.letter_stateseq), spkind_keys)[target_idx]
src_f0 = get_separated_values(np.load(args.f0), spkind_keys)[target_idx]
src_ap = get_separated_values(np.load(args.ap), spkind_keys)[target_idx]
src_mcep = get_separated_values(np.load(args.mcep), spkind_keys)[target_idx]

if args.sentences is None:
    if args.LM == "Unigram":
        snt_generator = Unigram_generator(args.sentences_file)
    elif args.LM == "Bigram":
        snt_generator = Bigram_generator(args.sentences_file, args.parameter)
    elif args.LM == "LSTM":
        snt_generator = LSTMLM_generator(args.LSTM_model, args.sentences_file)

ap_generator = AP_generator(args.letter_num, src_ap, letter_stateseq=src_letter_stateseq, flat=param_flat, mode=args.mode)
f0_generator = F0_generator(args.letter_num, src_f0, letter_stateseq=src_letter_stateseq, flat=param_flat, mode=args.mode)
feat_generator = NPBDAA_generator(args.parameter, mode=args.mode)

mcep_generator = MCEP_generator(args.letter_num, src_mcep, letter_stateseq=src_letter_stateseq, mode=args.mode)

if args.sentences is None:
    sentences = snt_generator.generate(size=args.size)
else:
    sentences = args.sentences
print(f"sentences: {sentences}")

for s, snt in enumerate(sentences):
    feature, gen_letter_stateseq = feat_generator.generate(snt)

    mcep = mcep_generator.generate(gen_letter_stateseq)
    ap = ap_generator.generate(gen_letter_stateseq)
    f0 = f0_generator.generate(gen_letter_stateseq)
    f0[f0<0] = 0

    mcep = signal.medfilt(mcep, (5, 1))
    mcep = mcep.astype(float, order="C")

    decoded_sp = pw.decode_spectral_envelope(mcep, args.samplerate, args.fftsize)
    synthesized = pw.synthesize(f0, decoded_sp, ap, args.samplerate, frame_period=args.frame_period*1000)
    synthesized = synthesized / max(abs(synthesized)) * 30000

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_file = args.output_prefix.with_name(f"{args.output_prefix.name}_{s:02d}_({'_'.join(map(str, snt))}).wav")

    wavfile.write(out_file, args.samplerate, synthesized.astype(np.int16))
