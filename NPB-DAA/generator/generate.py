import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import re
from math import floor
from scipy.stats import poisson, multivariate_normal

def load_parameters(param_file):
    param_npz = np.load(param_file)
    word_list = [tuple(param_npz[f"word_dicts/word({i})"]) for i in range(param_npz["num_states"])]
    phn_N = param_npz["letter_hsmm/num_states"]
    dur_lmbda = np.array([param_npz[f"letter_hsmm/dur_distns/dur_distn({i})/lmbda"] for i in range(phn_N)])
    obs_mu = np.array([param_npz[f"letter_hsmm/obs_distns/obs_distn({i})/mu"] for i in range(phn_N)])
    obs_sigma = np.array([param_npz[f"letter_hsmm/obs_distns/obs_distn({i})/sigma"] for i in range(phn_N)])

    return word_list, dur_lmbda, obs_mu, obs_sigma

class NPBDAA_generator(object):

    def __init__(self, word_list, dur_lmbda, obs_mu, obs_sigma, mode):
        assert mode in ["ML", "RND"]
        self.mode = mode
        self.word_list = word_list
        self.dur_lmbda = dur_lmbda
        self.obs_mu = obs_mu
        self.obs_sigma = obs_sigma
        self.phn_N = obs_mu.shape[0]
        self.obs_dim = obs_mu.shape[1]

    def __generate_dur(self, indexes):
        if self.mode == "ML":
            return np.array([floor(self.dur_lmbda[idx])+1 for idx in indexes])
        elif self.mode == "RND":
            return np.array([poisson.rvs(self.dur_lmbda[idx])+1 for idx in indexes])
        else:
            raise NotImplementedError(f"mode '{self.mode}' is not implemented")

    def __generate_obs(self, indexes, durations):
        T = np.sum(durations)
        D = self.obs_dim
        idx_ary = np.concatenate([np.full((dur, ), idx, dtype=int) for idx, dur in zip(indexes, durations)], axis=0)
        if self.mode == "ML":
            return self.obs_mu[idx_ary].copy()
        elif self.mode == "RND":
            out_matrix = np.empty((T, D))
            for idx, (mu, sigma) in enumerate(zip(self.obs_mu, self.obs_sigma)):
                out_matrix[idx_ary == idx] = multivariate_normal.rvs(mean=mu, cov=sigma, size=np.sum(idx_ary == idx))
            return out_matrix
        else:
            raise NotImplementedError(f"mode '{self.mode}' is not implemented")

    def generate(self, indexes):
        durs = self.__generate_dur(indexes)
        obs = self.__generate_obs(indexes, durs)
        return obs

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--parameter", type=Path, required=True)
parser.add_argument("--sentence", nargs="+", type=int, required=True)
parser.add_argument("--output", type=Path, default="generated_feature.txt")
parser.add_argument("--ML", action="store_true")

args = parser.parse_args()

# loading parameter
if args.ML:
    mode = "ML"
else:
    mode = "RND"

parameters = load_parameters(args.parameter)
g = NPBDAA_generator(*parameters, mode)

np.savetxt(args.output, g.generate(args.sentence))
