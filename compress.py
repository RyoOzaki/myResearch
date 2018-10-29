import numpy as np
from DSAE_PBHL.model import SAE, SAE_PBHL
from DSAE_PBHL.deep_model import DSAE, DSAE_PBHL
from DSAE_PBHL.util.normalizer import Normalizer

def packing(np_objs):
    return np.concatenate(np_objs, axis=0)

def packing_pb(np_objs, lengths, speaker_N, hot_val=1, cold_val=0):
    T = sum(lengths)
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    pb = np.ones((T, speaker_N)) * cold_val
    for i, id in enumerate(np_objs):
        pb[cumsum_lens[i]:cumsum_lens[i+1], id] = hot_val
    return pb

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

print("loading data...")
static = np.load("feature/static.npz")
delta = np.load("feature/delta.npz")
delta_delta = np.load("feature/delta_delta.npz")
speaker_id = np.load("feature/speaker_id.npz")

keys = list(speaker_id.keys())
lengths = [static[key].shape[0] for key in keys]
T = sum(lengths)
speaker_N = int(max([v for _,v in speaker_id.items()])) + 1

print("packing data...")
static_packed = packing([static[key] for key in keys])
delta_packed = packing([delta[key] for key in keys])
deltadelta_packed = packing([delta_delta[key] for key in keys])
speakerid_packed = packing_pb([speaker_id[key] for key in keys], lengths, speaker_N, cold_val=-1)

print("defining networks...")
structure = [[12, speaker_N], [8, int(speaker_N/10)]]

static_normalizer = Normalizer()
delta_normalizer = Normalizer()
deltadelta_normalizer = Normalizer()

# static_dsae_pbhl = DSAE_PBHL(structure)
# delta_dsae_pbhl = DSAE_PBHL(structure)
# deltadelta_dsae_pbhl = DSAE_PBHL(structure)
static_dsae_pbhl = SAE_PBHL(*structure)
delta_dsae_pbhl = SAE_PBHL(*structure)
deltadelta_dsae_pbhl = SAE_PBHL(*structure)

print("normalizing data...")
static_packed = static_normalizer.normalize(static_packed)
delta_packed = delta_normalizer.normalize(delta_packed)
deltadelta_packed = deltadelta_normalizer.normalize(deltadelta_packed)

static_normalizer.save_params("params/static_normalizer.npz")
delta_normalizer.save_params("params/delta_normalizer.npz")
deltadelta_normalizer.save_params("params/deltadelta_normalizer.npz")

print("training networks...")
static_dsae_pbhl.fit(static_packed, speakerid_packed)
static_dsae_pbhl.save_params("params/static_dsae_pbhl.npz")
delta_dsae_pbhl.fit(delta_packed, speakerid_packed)
delta_dsae_pbhl.save_params("params/delta_dsae_pbhl.npz")
deltadelta_dsae_pbhl.fit(deltadelta_packed, speakerid_packed)
deltadelta_dsae_pbhl.save_params("params/deltadelta_dsae_pbhl.npz")

static_packed = static_dsae_pbhl.feature(static_packed)
delta_packed = delta_dsae_pbhl.feature(delta_packed)
deltadelta_packed = deltadelta_dsae_pbhl.feature(deltadelta_packed)

print("unpacing data...")
static_unpacked = unpacking(static_packed, lengths)
delta_unpacked = unpacking(delta_packed, lengths)
deltadelta_unpacked = unpacking(deltadelta_packed, lengths)

print("making feature dict...")
static_compressed = {}
delta_compressed = {}
deltadelta_compressed = {}
for i, key in enumerate(keys):
    static_compressed[key] = static_unpacked[i]
    delta_compressed[key] = delta_unpacked[i]
    deltadelta_compressed[key] = deltadelta_unpacked[i]

print("saving data...")
np.savez("feature/static_compressed.npz", **static_compressed)
np.savez("feature/delta_compressed.npz", **delta_compressed)
np.savez("feature/delta_delta_compressed.npz", **deltadelta_compressed)

print("Finished!!")
