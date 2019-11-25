#%%
import numpy as np
from pyhlm.model import WeakLimitHDPHLM
from pyhlm.internals.hlm_states import WeakLimitHDPHLMStates
from pyhlm.word_model import LetterHSMM
import pyhsmm
import warnings
from tqdm import trange
warnings.filterwarnings('ignore')
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util.config_parser import ConfigParser_with_eval
from pathlib import Path
import pickle

#%% define functions
def load_config(file):
    cp = ConfigParser_with_eval()
    cp.read(file)
    return cp

def load_datas(train_data_file):
    train_data_npz = np.load(train_data_file)
    train_keys = sorted(list(train_data_npz.keys()))
    train_datas = [train_data_npz[key] for key in train_keys]
    return train_keys, train_datas

def unpack_durations(dur):
    unpacked = np.zeros(dur.sum())
    d = np.cumsum(dur[:-1])
    unpacked[d-1] = 1.0
    return unpacked

def save_model_as_pickle(iter_idx, model):
    with Path(f"models/ITR_{iter_idx:04d}.pickle").open(mode='wb') as file:
        pickle.dump(model, file)

def save_params_as_npz(iter_idx, model):
    params = model.params
    flatten_params = flatten_json(params)
    np.savez(f"parameters/ITR_{iter_idx:04d}.npz", **flatten_params)

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

def unflatten_json(flatten_json_obj):
    dict_obj = {}
    for keyname, value in flatten_json_obj.items():
        current_dict = dict_obj
        splitted_keyname = keyname.split("/")
        for key in splitted_keyname[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[splitted_keyname[-1]] = value
    return dict_obj

def copy_flatten_json(json_obj):
    new_json = {}
    for keyname, subjson in json_obj.items():
        type_of_subjson = type(subjson)
        if type_of_subjson in [int, float, complex, bool]:
            new_json[keyname] = subjson
        elif type_of_subjson in [list, tuple]:
            new_json[keyname] = subjson[:]
        elif type_of_subjson == np.ndarray:
            new_json[keyname] = subjson.copy()
        else:
            raise NotImplementedError(f"type :{type_of_subjson} can not copy. Plz implement here!")
    return new_json


#%% parse arguments
hypparams_model = "hypparams/model.config"
hypparams_letter_duration = "hypparams/letter_duration.config"
hypparams_letter_hsmm = "hypparams/letter_hsmm.config"
hypparams_letter_observation = "hypparams/letter_observation.config"
hypparams_pyhlm = "hypparams/pyhlm.config"
hypparams_word_length = "hypparams/word_length.config"

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--train_data", type=Path, required=True)

parser.add_argument("--model", default=hypparams_model, help="hyper parameters of model")
parser.add_argument("--letter_duration", type=Path, default=hypparams_letter_duration, help="hyper parameters of letter duration")
parser.add_argument("--letter_hsmm", type=Path, default=hypparams_letter_hsmm, help="hyper parameters of letter HSMM")
parser.add_argument("--letter_observation", type=Path, default=hypparams_letter_observation, help="hyper parameters of letter observation")
parser.add_argument("--pyhlm", type=Path, default=hypparams_pyhlm, help="hyper parameters of pyhlm")
parser.add_argument("--word_length", type=Path, default=hypparams_word_length, help="hyper parameters of word length")

args = parser.parse_args()

hypparams_model = args.model
hypparams_letter_duration = args.letter_duration
hypparams_letter_hsmm = args.letter_hsmm
hypparams_letter_observation = args.letter_observation
hypparams_pyhlm = args.pyhlm
hypparams_word_length = args.word_length

#%%
Path("results").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("parameters").mkdir(exist_ok=True)
Path("summary_files").mkdir(exist_ok=True)

#%% config parse
config_parser = load_config(hypparams_model)
section         = config_parser["model"]
thread_num      = section["thread_num"]
pretrain_iter   = section["pretrain_iter"]
train_iter      = section["train_iter"]
word_num        = section["word_num"]
letter_num      = section["letter_num"]
observation_dim = section["observation_dim"]
trunc = section["trunc"]


hlm_hypparams = load_config(hypparams_pyhlm)["pyhlm"]

config_parser = load_config(hypparams_letter_observation)
obs_hypparams = [config_parser[f"{i+1}_th"] for i in range(letter_num)]

config_parser = load_config(hypparams_letter_duration)
dur_hypparams = [config_parser[f"{i+1}_th"] for i in range(letter_num)]

len_hypparams = load_config(hypparams_word_length)["word_length"]

letter_hsmm_hypparams = load_config(hypparams_letter_hsmm)["letter_hsmm"]

#%% make instance of distributions and model
letter_obs_distns = [pyhsmm.distributions.Gaussian(**hypparam) for hypparam in obs_hypparams]
letter_dur_distns = [pyhsmm.distributions.PoissonDuration(**hypparam) for hypparam in dur_hypparams]
dur_distns = [pyhsmm.distributions.PoissonDuration(lmbda=20) for _ in range(word_num)]
length_distn = pyhsmm.distributions.PoissonDuration(**len_hypparams)

letter_hsmm = LetterHSMM(**letter_hsmm_hypparams, obs_distns=letter_obs_distns, dur_distns=letter_dur_distns)
model = WeakLimitHDPHLM(**hlm_hypparams, letter_hsmm=letter_hsmm, dur_distns=dur_distns, length_distn=length_distn)

#%%
train_keys, train_datas = load_datas(args.train_data)

#%% Pre training.
for data in train_datas:
    letter_hsmm.add_data(data, trunc=trunc)
for t in trange(pretrain_iter):
    letter_hsmm.resample_model(num_procs=thread_num)
letter_hsmm.states_list = []

#%%
print("Add datas...")
for data in train_datas:
    model.add_data(data, trunc=trunc, generate=False)
model.resample_states(num_procs=thread_num)
# # or
# for name, data in zip(files, train_datas):
#     model.add_data(data, trunc=trunc, initialize_from_prior=False)
print("Done!")

#%% alloc the memory of parameters and results
parameters_dicts = []
loglikelihoods = np.zeros(train_iter+1)
resample_times = np.zeros(train_iter)
word_stateseq = {key: np.zeros((train_iter, data.shape[0]), dtype=int) for key, data in zip(train_keys, train_datas)}
letter_stateseq = {key: np.zeros((train_iter, data.shape[0]), dtype=int) for key, data in zip(train_keys, train_datas)}
word_durations = {key: np.zeros((train_iter, data.shape[0]), dtype=int) for key, data in zip(train_keys, train_datas)}

#%% Save init params and models
loglikelihoods[0] = model.log_likelihood()
save_params_as_npz(0, model)
save_model_as_pickle(0, model)

#%%
for t in trange(train_iter):
    st = time.time()
    model.resample_model(num_procs=thread_num)
    resample_model_time = time.time() - st

    for key, states in zip(train_keys, model.states_list):
        word_stateseq[key][t] = states.stateseq.copy()
        letter_stateseq[key][t] = states.letter_stateseq.copy()
        word_durations[key][t] = unpack_durations(states.durations_censored)
    loglikelihoods[t+1] = model.log_likelihood()
    resample_times[t] = resample_model_time
    save_params_as_npz(t+1, model)
    save_model_as_pickle(t+1, model)
    print(f"log_likelihood:{model.log_likelihood()}")
    print(f"resample_model:{resample_model_time}")

np.savez("results/word_stateseq.npz", **word_stateseq)
np.savez("results/letter_stateseq.npz", **letter_stateseq)
np.savez("results/word_durations.npz", **word_durations)
np.savetxt("summary_files/log_likelihood.txt", loglikelihoods)
np.savetxt("summary_files/resample_times.txt", resample_times)
