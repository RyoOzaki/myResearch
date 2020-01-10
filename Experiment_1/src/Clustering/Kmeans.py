import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from sklearn.metrics import adjusted_rand_score

parser = ArgumentParser()

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--phn", type=Path, required=True)
parser.add_argument("--K", type=int, required=True)

args = parser.parse_args()

source = np.load(args.source)
phn = np.load(args.phn)
K = args.K

keys = sorted(list(source.keys()))
#
datas = np.concatenate([source[key] for key in keys], axis=0)
labels = np.concatenate([phn[key] for key in keys], axis=0)

N, D = datas.shape

means = np.empty((K, D))

lab = np.random.randint(K, size=(N,))
lab_cpy = np.full_like(lab, fill_value=-1, dtype=int)
distance = np.empty((N, K))

iter = 0
while np.any(lab != lab_cpy):
    lab_cpy = lab.copy()
    for k in range(K):
        kdata = datas[lab==k]
        if kdata.shape[0] > 0:
            means[k] = kdata.mean(axis=0)
        distance[:, k] = np.sum((datas - means[k])**2, axis=1)
    lab = np.argmin(distance, axis=1)
    # print(means)
    # sum_of_distance = 0
    # for k in range(K):
    #     sum_of_distance += distance[lab==k, k].sum()
    # print(sum_of_distance)
    iter += 1

for k in range(K):
    print(f"{k+1:2d}: {sum(lab==k)}")
print(f"ITR: {iter}")
print(f"ARI: {adjusted_rand_score(lab, labels)}")
