#%%
import numpy as np
from tqdm import trange, tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score, f1_score
from argparse import ArgumentParser
from util.config_parser import ConfigParser_with_eval
from pathlib import Path
import matplotlib.pyplot as plt

#%%
def load_config(filename):
    cp = ConfigParser_with_eval()
    cp.read(filename)
    return cp

def _convert_label(truth, predict, N):
    converted_label = np.full_like(truth, N)
    for true_lab in range(N):
        counted = [np.sum(predict[truth == true_lab] == pred) for pred in range(N)]
        pred_lab = np.argmax(counted)
        converted_label[predict == pred_lab] = true_lab
    return converted_label

def calc_f1_score(truth, predict, N, **kwargs):
    converted_predict = _convert_label(truth, predict, N)
    return f1_score(truth, converted_predict, labels=np.unique(converted_predict), **kwargs )

#%% parse arguments
default_hypparams_model = "hypparams/model.config"

parser = ArgumentParser()

parser.add_argument("--phn_label", type=Path)
parser.add_argument("--wrd_label", type=Path)

parser.add_argument("--model", default=default_hypparams_model, help=f"hyper parameters of model.")
args = parser.parse_args()

phn_labels = np.load(args.phn_label)
wrd_labels = np.load(args.wrd_label)
hypparams_model = args.model

Path("figures").mkdir(exist_ok=True)
Path("summary_files").mkdir(exist_ok=True)

#%% config parse
print("Loading model config...")
config_parser = load_config(hypparams_model)
section = config_parser["model"]
word_num = section["word_num"]
letter_num = section["letter_num"]
print("Done!")

#%%
print("Loading results....")
keys = sorted(list(phn_labels.keys()))
lengths = [phn_labels[key].shape[0] for key in keys]

word_results = np.load("results/word_stateseq.npz")
letter_results = np.load("results/letter_stateseq.npz")
duration_results = np.load("results/word_durations.npz")

cancatenated_phn_label = np.concatenate([phn_labels[key] for key in keys], axis=0)
cancatenated_wrd_label = np.concatenate([wrd_labels[key] for key in keys], axis=0)

concatenated_letter_result = np.concatenate([letter_results[key] for key in keys], axis=1)
concatenated_word_result = np.concatenate([word_results[key] for key in keys], axis=1)
# concatenated_duration_result = np.concatenate([duration_results[key] for key in keys], axis=1)

log_likelihood = np.loadtxt("summary_files/log_likelihood.txt")
resample_times = np.loadtxt("summary_files/resample_times.txt")
print("Done!")

train_iter = word_results[keys[0]].shape[0]

#%%
letter_ARI = np.zeros(train_iter)
letter_macro_f1_score = np.zeros(train_iter)
letter_micro_f1_score = np.zeros(train_iter)
word_ARI = np.zeros(train_iter)
word_macro_f1_score = np.zeros(train_iter)
word_micro_f1_score = np.zeros(train_iter)

#%% calculate ARI
print("Calculating ARI...")
for t in trange(train_iter):
    letter_ARI[t] = adjusted_rand_score(cancatenated_phn_label, concatenated_letter_result[t])
    letter_macro_f1_score[t] = calc_f1_score(cancatenated_phn_label, concatenated_letter_result[t], letter_num, average="macro")
    letter_micro_f1_score[t] = calc_f1_score(cancatenated_phn_label, concatenated_letter_result[t], letter_num, average="micro")
    word_ARI[t] = adjusted_rand_score(cancatenated_wrd_label, concatenated_word_result[t])
    word_macro_f1_score[t] = calc_f1_score(cancatenated_wrd_label, concatenated_word_result[t], word_num, average="macro")
    word_micro_f1_score[t] = calc_f1_score(cancatenated_wrd_label, concatenated_word_result[t], word_num, average="micro")
print("Done!")

#%%
print("Final ARI scores:")
print(f"Letter ARI: {letter_ARI[-1]}")
print(f"Word ARI: {word_ARI[-1]}")

#%% plot ARIs.
plt.clf()
plt.title("Letter ARI")
plt.plot(range(train_iter), letter_ARI, ".-")
plt.savefig("figures/Letter_ARI.png")

#%%
plt.clf()
plt.title("Word ARI")
plt.plot(range(train_iter), word_ARI, ".-")
plt.savefig("figures/Word_ARI.png")

#%%
plt.clf()
plt.title("Log likelihood")
plt.plot(range(train_iter+1), log_likelihood, ".-")
plt.savefig("figures/Log_likelihood.png")

#%%
plt.clf()
plt.title("Resample times")
plt.plot(range(train_iter), resample_times, ".-")
plt.savefig("figures/Resample_times.png")

#%%
np.savetxt("summary_files/Letter_ARI.txt", letter_ARI)
np.savetxt("summary_files/Letter_macro_F1_score.txt", letter_macro_f1_score)
np.savetxt("summary_files/Letter_micro_F1_score.txt", letter_micro_f1_score)
np.savetxt("summary_files/Word_ARI.txt", word_ARI)
np.savetxt("summary_files/Word_macro_F1_score.txt", word_macro_f1_score)
np.savetxt("summary_files/Word_micro_F1_score.txt", word_micro_f1_score)
np.savetxt("summary_files/Sum_of_resample_times.txt", np.array([resample_times.sum()]))
