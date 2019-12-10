import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from argparse import ArgumentParser

def packing(np_objs):
    return np.concatenate(np_objs, axis=0)

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

parser = ArgumentParser()

parser.add_argument("--source_file", type=Path, required=True)
parser.add_argument("--output_file", type=Path)

parser.add_argument("--n_components", type=int, required=True)

args = parser.parse_args()

out_file = args.output_file or args.source_file.with_name(f"compressed_{args.source_file.stem}_pca.npz")
out_file.parent.mkdir(exist_ok=True, parents=True)
param_dir = out_file.with_suffix("")
param_dir.mkdir(exist_ok=True)

npz_obj = np.load(args.source_file)
keys = sorted(list(npz_obj.keys()))
source_datas = [npz_obj[key] for key in keys]
lengths = [data.shape[0] for data in source_datas]
T = sum(lengths)

packed_source_datas = packing(source_datas)

pca = PCA(n_components=args.n_components)
pca.fit(packed_source_datas)
components = pca.components_
reduced_data = pca.transform(packed_source_datas)

unpacked_reduced_data = unpacking(reduced_data, lengths)
reduced = {}
for data, key in zip(unpacked_reduced_data, keys):
    reduced[key] = data

np.savez(out_file, **reduced)
np.save(param_dir / "components.npy", components)
