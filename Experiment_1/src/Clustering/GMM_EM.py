import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--n_components", type=int, required=True)
parser.add_argument("--trial", type=int, default=1)

args = parser.parse_args()

source = np.load(args.source)
phn = np.load(args.phn)

keys = sorted(list(source.keys()))

datas = np.concatenate([source[key] for key in keys], axis=0)
labels = np.concatenate([phn[key] for key in keys], axis=0)
N, D = datas.shape

aris = np.zeros(args.trial)
for t in range(args.trial):
    gmm = GaussianMixture(n_components=args.n_components, max_iter=1000)
    gmm.fit(datas)

    lab = gmm.predict(datas)

    aris[t] = adjusted_rand_score(lab, labels)

print(f"ARI: {aris}")
print(f"summary: {aris.mean()} +- {aris.std()}")
