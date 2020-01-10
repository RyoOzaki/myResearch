import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--n_components", type=int, required=True)

args = parser.parse_args()

source = np.load(args.source)
phn = np.load(args.phn)

keys = sorted(list(source.keys()))

datas = np.concatenate([source[key] for key in keys], axis=0)
labels = np.concatenate([phn[key] for key in keys], axis=0)
N, D = datas.shape

gmm = GaussianMixture(n_components=args.n_components, max_iter=1000)
gmm.fit(datas)

lab = gmm.predict(datas)

print(f"ARI: {adjusted_rand_score(lab, labels)}")
