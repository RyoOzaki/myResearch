import numpy as np
from LSTM_language_model.model import LSTM_language_model
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--train_data", type=Path, default="sentences.npz")

parser.add_argument("--separator", default=" ")
parser.add_argument("--size", type=int, default=1)
parser.add_argument("--with_BOS", action="store_true")
parser.add_argument("--with_EOS", action="store_true")

args = parser.parse_args()

sentences_npz = np.load(args.train_data)
depth = sentences_npz["depth"]
BOS_index = sentences_npz["BOS"]
EOS_index = sentences_npz["EOS"]

lstmlm = LSTM_language_model(depth, None, BOS_index=BOS_index, EOS_index=EOS_index, load_model_path=args.model)

sentences = lstmlm.generate(size=args.size)

if not args.with_BOS:
    sentences = [snt[1:] for snt in sentences]
if not args.with_EOS:
    sentences = [snt[:-1] for snt in sentences]

sentences = [args.separator.join(map(str, snt)) for snt in sentences]

for snt in sentences:
    print(snt)
