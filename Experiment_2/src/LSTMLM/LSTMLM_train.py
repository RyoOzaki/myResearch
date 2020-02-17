import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from LSTM_language_model.model import LSTM_language_model
from LSTM_language_model.utility import onehot, make_input_output

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output_model", type=Path)
parser.add_argument("--train_data", type=Path, default="sentences.npz")

parser.add_argument("--train_iter", type=int, default=100)
parser.add_argument("--hidden_node", type=int, default=128)

parser.add_argument("--init_model", action="store_true")

args = parser.parse_args()

sentences_npz = np.load(args.train_data)
depth = sentences_npz["depth"]
BOS_index = sentences_npz["BOS"]
EOS_index = sentences_npz["EOS"]

input_sentences, output_sentences = make_input_output(sentences_npz["sentences"])
input_matrix = onehot(input_sentences, depth=depth)
output_matrix = onehot(output_sentences, depth=depth)

output_model = args.output_model or args.model
output_model.parent.mkdir(parents=True, exist_ok=True)

if args.init_model:
    lstmlm = LSTM_language_model(depth, args.hidden_node, BOS_index=BOS_index, EOS_index=EOS_index)
else:
    lstmlm = LSTM_language_model(depth, None, BOS_index=BOS_index, EOS_index=EOS_index, load_model_path=args.model)

lstmlm.fit(
    input_matrix, output_matrix,
    epochs=args.train_iter,
)
lstmlm.save_model(output_model)
