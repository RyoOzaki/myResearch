import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--word_num", type=int, required=True)
parser.add_argument("--output", type=Path, default="sentences.npz")
parser.add_argument("--word_stateseq", type=Path, required=True)
parser.add_argument("--word_durations", type=Path, required=True)
parser.add_argument("--using_iterations", nargs="+", type=int, default=[-1, -2, -3])

args = parser.parse_args()

masking_value = -1
word_num = args.word_num
BOS_index = word_num
EOS_index = word_num + 1

# load results
word_stateseq = np.load(args.word_stateseq)
word_durations = np.load(args.word_durations)
keys = list(word_stateseq.keys())
sentences = [
    word_stateseq[key][iter, word_durations[key][iter] == 1]
    for key in keys for iter in args.using_iterations
]
max_length = max(map(lambda x: x.shape[0], sentences))

# make sentence_matrix
sentence_matrix = np.full((len(sentences), max_length+2), fill_value=masking_value, dtype=int)

sentence_matrix[:, 0] = BOS_index
for i, snt in enumerate(sentences):
    sentence_matrix[i, 1:snt.shape[0]+1] = snt
    sentence_matrix[i, snt.shape[0]+1] = EOS_index

np.savez(args.output,
    shape=sentence_matrix.shape,
    depth=word_num+2,
    BOS=BOS_index,
    EOS=EOS_index,
    sentences=sentence_matrix
)
