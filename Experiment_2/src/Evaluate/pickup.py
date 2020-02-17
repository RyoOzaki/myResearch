import numpy as np
from pathlib import Path
from argparse import ArgumentParser

def separate_speaker(speaker_npz_obj):
    all_speaker = sorted(list(set(map(str, speaker_npz_obj.values()))))
    all_keys = sorted(list(speaker_npz_obj.keys()))
    speaker_individual_keys = [
        [
            key
        for key in all_keys if speaker_npz_obj[key] == speaker
        ]
    for speaker in all_speaker
    ]
    return all_speaker, speaker_individual_keys

parser = ArgumentParser()

parser.add_argument("--raw_sentence", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)

parser.add_argument("--sentence", nargs="+", type=str)

parser.add_argument("--pattern", type=str, default="{speaker}/{sentence}")
parser.add_argument("--format", type=str, default="--sentences\n{sentence}")

parser.add_argument("--output_file", type=Path, required=True)
parser.add_argument("--key_output", type=Path, required=True)

args = parser.parse_args()
args.output_file.parent.mkdir(exist_ok=True, parents=True)

raw_sentence_npz = np.load(args.raw_sentence)
speakers, _ = separate_speaker(np.load(args.speaker_id))

output_strs = []
keys = []
for spk in speakers:
    for snt in args.sentence:
        key = args.pattern.format(speaker=spk, sentence=snt)
        keys.append(key)
        output_strs.append(args.format.format(sentence="\n".join(map(str, list(raw_sentence_npz[key])))))
args.output_file.write_text("\n".join(output_strs))

args.key_output.parent.mkdir(exist_ok=True, parents=True)
args.key_output.write_text("\n".join(keys))
