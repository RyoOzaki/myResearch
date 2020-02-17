import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt

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

parser.add_argument("--word_num", type=int, required=True)
parser.add_argument("--raw_sentence", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)

parser.add_argument("--output_dir", type=Path, default="./figure/")
parser.add_argument("--cmap", default="tab10")

parser.add_argument("--format", default="png")

args = parser.parse_args()

args.output_dir.mkdir(exist_ok=True, parents=True)

cmap = plt.get_cmap(args.cmap)

raw_sentence_npz = np.load(args.raw_sentence)
if args.speaker_id is None:
    wrd_cnt = np.zeros((args.word_num, ), dtype=int)
    for key, val in row_sentence_npz.items():
        for v in val:
            wrd_cnt[int(v)] += 1
else:
    speakers, spkind_keys = separate_speaker(np.load(args.speaker_id))
    speaker_num = len(speakers)
    spkind_wrd_cnt = np.zeros((speaker_num, args.word_num), dtype=int)
    for spk_idx, spk_keys in enumerate(spkind_keys):
        for key in spk_keys:
            for v in raw_sentence_npz[key]:
                spkind_wrd_cnt[spk_idx, int(v)] += 1

    cum_spkind_wrd_cnt = np.cumsum(spkind_wrd_cnt, axis=1)
    cum_spkind_wrd_cnt = np.concatenate((np.zeros((speaker_num, 1), dtype=int), cum_spkind_wrd_cnt), axis=1)
    x = np.arange(speaker_num)
    for wrd in range(args.word_num):
        plt.bar(x, spkind_wrd_cnt[:, wrd], bottom=cum_spkind_wrd_cnt[:, wrd], width=0.8, color=cmap(wrd), label=f"word {wrd}")
    xlim = plt.xlim()
    plt.xlim((xlim[0], xlim[1]+1))
    plt.xticks(x, speakers)
    plt.xlabel("Speakers")
    plt.ylabel("Word count")
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(args.output_dir / f"spkind_word_count.{args.format}")

    wrd_cnt = spkind_wrd_cnt.sum(axis=0)

x = np.arange(args.word_num)
plt.clf()
plt.bar(x, wrd_cnt)
xticks = [f"{i}" for i in range(args.word_num)]
plt.xticks(x, xticks)
plt.xlabel("Word index")
plt.ylabel("Word count")
plt.tight_layout()
# plt.legend()
plt.savefig(args.output_dir / f"word_count.{args.format}")
