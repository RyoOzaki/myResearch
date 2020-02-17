import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import sys
import re
from math import floor
from scipy.stats import poisson, multivariate_normal

def load_NPBDAA_parameters(param_file):
    param_npz = np.load(param_file)
    word_list = [tuple(param_npz[f"word_dicts/word({i})"]) for i in range(param_npz["num_states"])]
    phn_N = param_npz["letter_hsmm/num_states"]
    dur_lmbda = np.array([param_npz[f"letter_hsmm/dur_distns/dur_distn({i})/lmbda"] for i in range(phn_N)])
    obs_mu = np.array([param_npz[f"letter_hsmm/obs_distns/obs_distn({i})/mu"] for i in range(phn_N)])
    obs_sigma = np.array([param_npz[f"letter_hsmm/obs_distns/obs_distn({i})/sigma"] for i in range(phn_N)])

    return word_list, dur_lmbda, obs_mu, obs_sigma

def conv_to_letter_stateseq_norep(word_list, word_stateseq_norep):
    letter_stateseq_norep = np.concatenate([word_list[w] for w in word_stateseq_norep])
    return letter_stateseq_norep

class NPBDAA_generator(object):

    def __init__(self, parameter_file, mode):
        params = load_NPBDAA_parameters(parameter_file)
        self.__set_params(*params, mode)

    def __set_params(self, word_list, dur_lmbda, obs_mu, obs_sigma, mode):
        assert mode in ["ML", "RND"]
        self.mode = mode
        self.word_list = word_list
        self.dur_lmbda = dur_lmbda
        self.obs_mu = obs_mu
        self.obs_sigma = obs_sigma
        self.phn_N = obs_mu.shape[0]
        self.obs_dim = obs_mu.shape[1]

    def __generate_dur(self, letter_stateseq_norep):
        if self.mode == "ML":
            return np.array([floor(self.dur_lmbda[idx])+1 for idx in letter_stateseq_norep])
        elif self.mode == "RND":
            return np.array([poisson.rvs(self.dur_lmbda[idx])+1 for idx in letter_stateseq_norep])
        else:
            raise NotImplementedError(f"mode '{self.mode}' is not implemented")

    def __generate_obs(self, letter_stateseq_norep, durations):
        T = np.sum(durations)
        D = self.obs_dim
        idx_ary = np.concatenate([np.full((dur, ), idx, dtype=int) for idx, dur in zip(letter_stateseq_norep, durations)], axis=0)
        if self.mode == "ML":
            return self.obs_mu[idx_ary].copy(), idx_ary
        elif self.mode == "RND":
            out_matrix = np.empty((T, D))
            for idx, (mu, sigma) in enumerate(zip(self.obs_mu, self.obs_sigma)):
                out_matrix[idx_ary == idx] = multivariate_normal.rvs(mean=mu, cov=sigma, size=np.sum(idx_ary == idx))
            return out_matrix, idx_ary
        else:
            raise NotImplementedError(f"mode '{self.mode}' is not implemented")

    def generate(self, word_stateseq_norep):
        letter_stateseq_norep = conv_to_letter_stateseq_norep(self.word_list, word_stateseq_norep)
        durs = self.__generate_dur(letter_stateseq_norep)
        obs, letter_stateseq = self.__generate_obs(letter_stateseq_norep, durs)
        return obs, letter_stateseq
