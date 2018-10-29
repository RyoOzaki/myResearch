import numpy as np

static = np.load("feature/static_compressed.npz")
delta = np.load("feature/delta_compressed.npz")
delta_delta = np.load("feature/delta_delta_compressed.npz")

data = {}
for key in static.keys():
    s = static[key]
    d = delta[key]
    dd = delta_delta[key]
    data[key] = np.concatenate((s, d, dd), axis=1)

np.savez("feature/concat_data.npz", **data)
