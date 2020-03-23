#%%
import numpy as np
from tqdm import trange, tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import adjusted_rand_score, f1_score
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import product
from util.utility import separate_speaker, get_separated_values

from configparser import ConfigParser

class ConfigParser_with_eval(ConfigParser):
    def get(self, *argv, **kwargs):
        import numpy
        val = super(ConfigParser_with_eval, self).get(*argv, **kwargs)
        return eval(val)

#%%
def load_config(filename):
    cp = ConfigParser_with_eval()
    cp.read(filename)
    return cp

def make_confusion_matrix(truth, predict, truth_N, predict_N):
    cm = np.zeros((truth_N, predict_N), dtype=int)
    for t in range(truth_N):
        cm[t] = np.bincount(predict[truth == t], minlength=predict_N)
    return cm

def convert_label(predict, confusion_matrix):
    truth_N, predict_N = confusion_matrix.shape
    mapping_labels = np.argmax(confusion_matrix, axis=1)
    converted_label = np.full_like(predict, predict_N)
    for t in range(truth_N):
        converted_label[predict == mapping_labels[t]] = t
    return converted_label

def calc_f1_score(truth, predict, confusion_matrix, **kwargs):
    converted_predict = convert_label(predict, confusion_matrix)
    return f1_score(truth, converted_predict, labels=np.unique(converted_predict), **kwargs)

def calc_scores(truth, predict, truth_N, predict_N):
    ARI = adjusted_rand_score(truth, predict)
    confusion_matrix = make_confusion_matrix(truth, predict, truth_N, predict_N)
    macro_F1 = calc_f1_score(truth, predict, confusion_matrix, average="macro")
    micro_F1 = calc_f1_score(truth, predict, confusion_matrix, average="micro")
    return ARI, macro_F1, micro_F1, confusion_matrix

#%% parse arguments
default_hypparams_model = "hypparams/model.config"

parser = ArgumentParser()

parser.add_argument("--phn_label", type=Path, required=True)
parser.add_argument("--wrd_label", type=Path, required=True)
parser.add_argument("--speaker_id", type=Path)

parser.add_argument("--model", default=default_hypparams_model, help=f"hyper parameters of model.")

parser.add_argument("--figure_dir", type=Path, default="./figures")
parser.add_argument("--summary_dir", type=Path, default="./summary_files")
parser.add_argument("--results_dir", type=Path, default="./results")

args = parser.parse_args()

phn_labels = np.load(args.phn_label)
phn_label_N = int(max(map(max, phn_labels.values())) + 1)
wrd_labels = np.load(args.wrd_label)
wrd_label_N = int(max(map(max, wrd_labels.values())) + 1)

hypparams_model = args.model

figure_dir = args.figure_dir
figure_dir.mkdir(exist_ok=True, parents=True)
summary_dir = args.summary_dir
summary_dir.mkdir(exist_ok=True, parents=True)
results_dir = args.results_dir

#%% config parse
print("Loading model config...")
config_parser = load_config(hypparams_model)
section = config_parser["model"]
word_num = section["word_num"]
letter_num = section["letter_num"]
print("Done!")

#%%
print("Loading results....")
word_results = np.load(results_dir / "word_stateseq.npz")
letter_results = np.load(results_dir / "letter_stateseq.npz")
duration_results = np.load(results_dir / "word_durations.npz")
keys = sorted(list(word_results.keys()))
train_iter = word_results[keys[0]].shape[0]

if args.speaker_id is not None:
    speaker, spkind_keys = separate_speaker(np.load(args.speaker_id))
    speaker_N = len(speaker)
    spkind_phn_labels = get_separated_values(phn_labels, spkind_keys)
    spkind_wrd_labels = get_separated_values(wrd_labels, spkind_keys)
    spkind_letter_results = get_separated_values(letter_results, spkind_keys)
    spkind_word_results = get_separated_values(word_results, spkind_keys)

    spkind_letter_ARI = np.zeros((speaker_N, train_iter))
    spkind_letter_macro_f1_score = np.zeros((speaker_N, train_iter))
    spkind_letter_micro_f1_score = np.zeros((speaker_N, train_iter))
    spkind_word_ARI = np.zeros((speaker_N, train_iter))
    spkind_word_macro_f1_score = np.zeros((speaker_N, train_iter))
    spkind_word_micro_f1_score = np.zeros((speaker_N, train_iter))

    spkind_letter_confusion_matrix = np.zeros((speaker_N, train_iter, phn_label_N, letter_num), dtype=int)
    spkind_word_confusion_matrix = np.zeros((speaker_N, train_iter, wrd_label_N, word_num), dtype=int)
    for spk_idx, spk in enumerate(speaker):
        concatenated_phn_label = np.concatenate(spkind_phn_labels[spk_idx], axis=0)
        concatenated_letter_result = np.concatenate(spkind_letter_results[spk_idx], axis=1)
        concatenated_wrd_label = np.concatenate(spkind_wrd_labels[spk_idx], axis=0)
        concatenated_word_result = np.concatenate(spkind_word_results[spk_idx], axis=1)
        for t in range(train_iter):
            ARI, macro_F1, micro_F1, confusion_matrix = calc_scores(
                concatenated_phn_label,
                concatenated_letter_result[t],
                phn_label_N,
                letter_num
            )
            spkind_letter_ARI[spk_idx, t] = ARI
            spkind_letter_macro_f1_score[spk_idx, t] = macro_F1
            spkind_letter_micro_f1_score[spk_idx, t] = micro_F1
            spkind_letter_confusion_matrix[spk_idx, t] = confusion_matrix

            ARI, macro_F1, micro_F1, confusion_matrix = calc_scores(
                concatenated_wrd_label,
                concatenated_word_result[t],
                wrd_label_N,
                word_num
            )
            spkind_word_ARI[spk_idx, t] = ARI
            spkind_word_macro_f1_score[spk_idx, t] = macro_F1
            spkind_word_micro_f1_score[spk_idx, t] = micro_F1
            spkind_word_confusion_matrix[spk_idx, t] = confusion_matrix

    # plot and save datas
    plt.clf()
    plt.title("Letter ARI")
    for spk_idx, spk in enumerate(speaker):
        plt.plot(range(train_iter), spkind_letter_ARI[spk_idx], ".-", label=spk)
    plt.legend()
    plt.savefig(figure_dir / "Speaker_individual_letter_ARI.png")

    plt.clf()
    plt.title("Word ARI")
    for spk_idx, spk in enumerate(speaker):
        plt.plot(range(train_iter), spkind_word_ARI[spk_idx], ".-", label=spk)
    plt.legend()
    plt.savefig(figure_dir / "Speaker_individual_word_ARI.png")

    truth_num = spkind_letter_confusion_matrix.shape[2]
    pred_num = spkind_letter_confusion_matrix.shape[3]
    out_fig_dir = figure_dir / "Speaker_individual_letter_confusion_matrix"
    out_fig_dir.mkdir(exist_ok=True)
    for spk_idx, spk in enumerate(speaker):
        plt.clf()
        plt.title(f"Letter confusion matrix of {spk}")
        plt.imshow(spkind_letter_confusion_matrix[spk_idx, -1])
        for i, j in product(range(truth_num), range(pred_num)):
            if spkind_letter_confusion_matrix[spk_idx, -1, i, j] != 0:
                plt.text(j, i, spkind_letter_confusion_matrix[spk_idx, -1, i, j], ha="center", va="center", color="white")
        plt.yticks(np.arange(truth_num), map(lambda x: f"{x:2d}", np.arange(truth_num)))
        plt.xticks(np.arange(pred_num), map(lambda x: f"{x:2d}", np.arange(pred_num)))
        plt.ylabel("Truth index")
        plt.xlabel("Predict index")
        plt.savefig(out_fig_dir / f"{spk}.png")

    truth_num = spkind_word_confusion_matrix.shape[2]
    pred_num = spkind_word_confusion_matrix.shape[3]
    out_fig_dir = figure_dir / "Speaker_individual_word_confusion_matrix"
    out_fig_dir.mkdir(exist_ok=True)
    for spk_idx, spk in enumerate(speaker):
        plt.clf()
        plt.title(f"Word confusion matrix of {spk}")
        plt.imshow(spkind_word_confusion_matrix[spk_idx, -1])
        for i, j in product(range(truth_num), range(pred_num)):
            if spkind_word_confusion_matrix[spk_idx, -1, i, j] != 0:
                plt.text(j, i, spkind_word_confusion_matrix[spk_idx, -1, i, j], ha="center", va="center", color="white")
        plt.yticks(np.arange(truth_num), map(lambda x: f"{x:2d}", np.arange(truth_num)))
        plt.xticks(np.arange(pred_num), map(lambda x: f"{x:2d}", np.arange(pred_num)))
        plt.ylabel("Truth index")
        plt.xlabel("Predict index")
        plt.savefig(out_fig_dir / f"{spk}.png")
    np.savetxt(summary_dir / "Speaker_individual_letter_ARI.txt", spkind_letter_ARI)
    np.savetxt(summary_dir / "Speaker_individual_letter_macro_F1_score.txt", spkind_letter_macro_f1_score)
    np.savetxt(summary_dir / "Speaker_individual_letter_micro_F1_score.txt", spkind_letter_micro_f1_score)
    np.save(summary_dir / "Speaker_individual_letter_confusion_matrix.npy", spkind_letter_confusion_matrix)
    np.savetxt(summary_dir / "Speaker_individual_word_ARI.txt", spkind_word_ARI)
    np.savetxt(summary_dir / "Speaker_individual_word_macro_F1_score.txt", spkind_word_macro_f1_score)
    np.savetxt(summary_dir / "Speaker_individual_word_micro_F1_score.txt", spkind_word_micro_f1_score)
    np.save(summary_dir / "Speaker_individual_word_confusion_matrix.npy", spkind_word_confusion_matrix)

# For all speaker
concatenated_phn_label = np.concatenate([phn_labels[key] for key in keys], axis=0)
concatenated_wrd_label = np.concatenate([wrd_labels[key] for key in keys], axis=0)

concatenated_letter_result = np.concatenate([letter_results[key] for key in keys], axis=1)
concatenated_word_result = np.concatenate([word_results[key] for key in keys], axis=1)
# concatenated_duration_result = np.concatenate([duration_results[key] for key in keys], axis=1)

log_likelihood = np.loadtxt(summary_dir / "log_likelihood.txt")
resample_times = np.loadtxt(summary_dir / "resample_times.txt")
print("Done!")

#%%
letter_ARI = np.zeros(train_iter)
letter_macro_f1_score = np.zeros(train_iter)
letter_micro_f1_score = np.zeros(train_iter)
letter_confusion_matrix = np.zeros((train_iter, phn_label_N, letter_num), dtype=int)
word_ARI = np.zeros(train_iter)
word_macro_f1_score = np.zeros(train_iter)
word_micro_f1_score = np.zeros(train_iter)
word_confusion_matrix = np.zeros((train_iter, wrd_label_N, word_num), dtype=int)

#%% calculate ARI
print("Calculating ARI...")
for t in trange(train_iter):
    ARI, macro_F1, micro_F1, confusion_matrix = calc_scores(
        concatenated_phn_label,
        concatenated_letter_result[t],
        phn_label_N,
        letter_num
    )
    letter_ARI[t] = ARI
    letter_macro_f1_score[t] = macro_F1
    letter_micro_f1_score[t] = micro_F1
    letter_confusion_matrix[t] = confusion_matrix

    ARI, macro_F1, micro_F1, confusion_matrix = calc_scores(
        concatenated_wrd_label,
        concatenated_word_result[t],
        wrd_label_N,
        word_num
    )
    word_ARI[t] = ARI
    word_macro_f1_score[t] = macro_F1
    word_micro_f1_score[t] = micro_F1
    word_confusion_matrix[t] = confusion_matrix
print("Done!")

#%%
print("Final ARI scores:")
print(f"Letter ARI: {letter_ARI[-1]}")
print(f"Word ARI: {word_ARI[-1]}")

#%% plot ARIs.
plt.clf()
plt.title("Letter ARI")
plt.plot(range(train_iter), letter_ARI, ".-")
plt.savefig(figure_dir / "Letter_ARI.png")

#%%
plt.clf()
plt.title("Word ARI")
plt.plot(range(train_iter), word_ARI, ".-")
plt.savefig(figure_dir / "Word_ARI.png")

#%%
plt.clf()
plt.title("Log likelihood")
plt.plot(range(train_iter+1), log_likelihood, ".-")
plt.savefig(figure_dir / "Log_likelihood.png")

#%%
plt.clf()
plt.title("Resample times")
plt.plot(range(train_iter), resample_times, ".-")
plt.savefig(figure_dir / "Resample_times.png")

plt.clf()
plt.title(f"Word confusion matrix")
plt.imshow(word_confusion_matrix[-1])
for i, j in product(range(truth_num), range(pred_num)):
    if word_confusion_matrix[-1, i, j] != 0:
        plt.text(j, i, word_confusion_matrix[-1, i, j], ha="center", va="center", color="white")
plt.yticks(np.arange(truth_num), map(lambda x: f"{x:2d}", np.arange(truth_num)))
plt.xticks(np.arange(pred_num), map(lambda x: f"{x:2d}", np.arange(pred_num)))
plt.ylabel("Truth index")
plt.xlabel("Predict index")
plt.savefig(figure_dir / f"Word_confusion_matrix.png")

plt.clf()
plt.title(f"Letter confusion matrix")
plt.imshow(letter_confusion_matrix[-1])
for i, j in product(range(truth_num), range(pred_num)):
    if letter_confusion_matrix[-1, i, j] != 0:
        plt.text(j, i, letter_confusion_matrix[-1, i, j], ha="center", va="center", color="white")
plt.yticks(np.arange(truth_num), map(lambda x: f"{x:2d}", np.arange(truth_num)))
plt.xticks(np.arange(pred_num), map(lambda x: f"{x:2d}", np.arange(pred_num)))
plt.ylabel("Truth index")
plt.xlabel("Predict index")
plt.savefig(figure_dir / f"Letter_confusion_matrix.png")

np.savetxt(summary_dir / "Letter_ARI.txt", letter_ARI)
np.savetxt(summary_dir / "Letter_macro_F1_score.txt", letter_macro_f1_score)
np.savetxt(summary_dir / "Letter_micro_F1_score.txt", letter_micro_f1_score)
np.save(summary_dir / "Letter_confusion_matrix.npy", letter_confusion_matrix)
np.savetxt(summary_dir / "Word_ARI.txt", word_ARI)
np.savetxt(summary_dir / "Word_macro_F1_score.txt", word_macro_f1_score)
np.savetxt(summary_dir / "Word_micro_F1_score.txt", word_micro_f1_score)
np.save(summary_dir / "Word_confusion_matrix.npy", word_confusion_matrix)
np.savetxt(summary_dir / "Sum_of_resample_times.txt", np.array([resample_times.sum()]))
