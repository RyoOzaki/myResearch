import numpy as np
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from DSAE_PBHL import AE, SAE, SAE_PBHL
from DSAE_PBHL import DSAE, DSAE_PBHL
from DSAE_PBHL.util import Builder

def flatten_json(json_obj, keyname_prefix=None, dict_obj=None):
    if dict_obj is None:
        dict_obj = {}
    if keyname_prefix is None:
        keyname_prefix = ""
    for keyname, subjson in json_obj.items():
        if type(subjson) == dict:
            prefix = f"{keyname_prefix}{keyname}/"
            flatten_json(subjson, keyname_prefix=prefix, dict_obj=dict_obj)
        else:
            dict_obj[f"{keyname_prefix}{keyname}"] = subjson
    return dict_obj

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

# make onehot
def packing_pb(label_objs, lengths, speaker_N):
    T = sum(lengths)
    id_array = np.concatenate([np.full(t, id) for t, id in zip(lengths, label_objs)])
    return np.identity(speaker_N)[id_array]

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--train_data", type=Path, required=True)
parser.add_argument("--output", type=Path)
parser.add_argument("--speaker_id", type=Path, required=True)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--threshold", type=float, default=1.0E-60)
# parser.add_argument("--model", type=Path)
parser.add_argument("--structure", type=int, nargs="+", required=True)
parser.add_argument("--pb_structure", type=int, nargs=2, required=True)

args = parser.parse_args()

print("loading data...")
npz_obj = np.load(args.train_data)
speaker_npz_obj = np.load(args.speaker_id)
keys = sorted(list(npz_obj.keys()))
all_speakers = sorted(list(set(map(str, speaker_npz_obj.values()))))

train_datas = [npz_obj[key] for key in keys]
speaker_ids = [all_speakers.index(speaker_npz_obj[key]) for key in keys]
speaker_N = len(all_speakers)

print("packing data...")
packed_train_datas, lengths = packing(train_datas)
packed_speaker_ids = packing_pb(speaker_ids, lengths, speaker_N)

print("defining networks...")
structure = args.structure
pb_structure = args.pb_structure
L = len(structure)
builder = Builder(structure[0], pb_input_dim=pb_structure[0])
for dim in structure[1:-1]:
    builder.stack(SAE, dim)
builder.stack(SAE_PBHL, structure[-1], pb_hidden_dim=pb_structure[1])
builder.print_recipe()

with tf.variable_scope("dsae_pbhl"):
    dsae = builder.build()

print("training networks...")
epoch = args.epoch
threshold = args.threshold
saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
with tf.Session() as sess:
    initializer = tf.global_variables_initializer()
    sess.run(initializer)
    for i in range(L-1):
        print(f"Training {i+1} th network (all:{L-1})")
        dsae.fit_until(sess, i, packed_train_datas, packed_speaker_ids, epoch, threshold)
    # saver.save(sess, model_ckpt, global_step=1)
    compressed = dsae.hidden_layers_with_eval(sess, packed_train_datas)[-1]
    compressed_pb = dsae.hidden_layer_pb_with_eval(sess, packed_train_datas, packed_speaker_ids)
    dsae_pbhl_params = sess.run(dsae.params)

print("unpacing data...")
unpacked = unpacking(compressed, lengths)
unpacked_pb = unpacking(compressed_pb, lengths)

print("making feature dict...")
compressed = {}
for data, key in zip(unpacked, keys):
    compressed[key] = data
compressed_pb = {}
for data, key in zip(unpacked_pb, keys):
    compressed_pb[key] = data
pb_means = {}
for speaker in all_speakers:
    spk_keys = [key for key in keys if speaker_npz_obj[key] == speaker]
    concat_pb = np.concatenate([compressed_pb[key] for key in spk_keys], axis=0)
    pb_means[speaker] = concat_pb.mean(axis=0)

print("saving data...")
out_file = args.output or args.train_data.with_name(f"dsae_pbhl_{args.train_data.stem}.npz")
out_file.parent.mkdir(exist_ok=True, parents=True)
param_dir = out_file.with_suffix("")
param_dir.mkdir(exist_ok=True)
np.savez(out_file, **compressed)
np.savez(param_dir / "dsae.npz", **flatten_json(dsae_pbhl_params))
np.savez(param_dir / "pb_means.npz", **pb_means)

print("Finished!!")
