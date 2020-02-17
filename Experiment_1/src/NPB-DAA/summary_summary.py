#%%
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import re

#%%
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--result_dir", type=Path, required=True)

parser.add_argument("--figure_dir", type=Path, default="./figures")
parser.add_argument("--summary_dir", type=Path, default="./summary_files")

args = parser.parse_args()

#%%
dirs = [dir for dir in args.result_dir.iterdir() if dir.is_dir() and re.match(r"^[0-9]+$", dir.stem)]
dirs.sort(key=lambda dir: dir.stem)

#%%
figure_dir = args.figure_dir
figure_dir.mkdir(exist_ok=True, parents=True)
summary_dir = args.summary_dir
summary_dir.mkdir(exist_ok=True, parents=True)

#%%
print("Initialize variables....")
N = len(dirs)
tmp = np.loadtxt(dirs[0] / "summary_files/resample_times.txt")
T = tmp.shape[0]
spkind = False
if (dirs[0] / "summary_files/Speaker_individual_letter_ARI.txt").exists():
    tmp = np.loadtxt(dirs[0] / "summary_files/Speaker_individual_letter_ARI.txt")
    spk_num = tmp.shape[0]
    spkind = True

resample_times = np.empty((N, T))
log_likelihoods = np.empty((N, T+1))

letter_ARIs = np.empty((N, T))
letter_macro_f1_scores = np.empty((N, T))
letter_micro_f1_scores = np.empty((N, T))

word_ARIs = np.empty((N, T))
word_macro_f1_scores = np.empty((N, T))
word_micro_f1_scores = np.empty((N, T))

if spkind:
    spkind_letter_ARIs = np.empty((N, spk_num, T))
    spkind_word_ARIs = np.empty((N, spk_num, T))

print("Done!")

#%%
print("Loading results....")
for i, dir in enumerate(dirs):
    resample_times[i] = np.loadtxt(dir / "summary_files/resample_times.txt")
    log_likelihoods[i] = np.loadtxt(dir / "summary_files/log_likelihood.txt")
    letter_ARIs[i] = np.loadtxt(dir / "summary_files/Letter_ARI.txt")
    letter_macro_f1_scores[i] = np.loadtxt(dir / "summary_files/Letter_macro_F1_score.txt")
    letter_micro_f1_scores[i] = np.loadtxt(dir / "summary_files/Letter_micro_F1_score.txt")
    word_ARIs[i] = np.loadtxt(dir / "summary_files/Word_ARI.txt")
    word_macro_f1_scores[i] = np.loadtxt(dir / "summary_files/Word_macro_F1_score.txt")
    word_micro_f1_scores[i] = np.loadtxt(dir / "summary_files/Word_micro_F1_score.txt")

    if spkind:
        spkind_letter_ARIs[i] = np.loadtxt(dir / "summary_files/Speaker_individual_letter_ARI.txt")
        spkind_word_ARIs[i] = np.loadtxt(dir / "summary_files/Speaker_individual_word_ARI.txt")

print("Done!")

#%%
print("Ploting...")
plt.clf()
plt.errorbar(range(T), resample_times.mean(axis=0), yerr=resample_times.std(axis=0))
plt.xlabel("Iteration")
plt.ylabel("Execution time [sec]")
plt.title("Transitions of the execution time")
plt.savefig(figure_dir / "summary_of_execution_time.png")

plt.clf()
plt.errorbar(range(T+1), log_likelihoods.mean(axis=0), yerr=log_likelihoods.std(axis=0))
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title("Transitions of the log likelihood")
plt.savefig(figure_dir / "summary_of_log_likelihood.png")

plt.clf()
plt.errorbar(range(T), word_ARIs.mean(axis=0), yerr=word_ARIs.std(axis=0), label="Word ARI")
plt.errorbar(range(T), letter_ARIs.mean(axis=0), yerr=letter_ARIs.std(axis=0), label="Letter ARI")
plt.xlabel("Iteration")
plt.ylabel("ARI")
plt.title("Transitions of the ARI")
plt.legend()
plt.savefig(figure_dir / "summary_of_ARI.png")

plt.clf()
plt.errorbar(range(T), word_macro_f1_scores.mean(axis=0), yerr=word_macro_f1_scores.std(axis=0), label="Word macro F1")
plt.errorbar(range(T), letter_macro_f1_scores.mean(axis=0), yerr=letter_macro_f1_scores.std(axis=0), label="Letter macro F1")
plt.xlabel("Iteration")
plt.ylabel("Macro F1 score")
plt.title("Transitions of the macro F1 score")
plt.legend()
plt.savefig(figure_dir / "summary_of_macro_F1_score.png")

plt.clf()
plt.errorbar(range(T), word_micro_f1_scores.mean(axis=0), yerr=word_micro_f1_scores.std(axis=0), label="Word micro F1")
plt.errorbar(range(T), letter_micro_f1_scores.mean(axis=0), yerr=letter_micro_f1_scores.std(axis=0), label="Letter micro F1")
plt.xlabel("Iteration")
plt.ylabel("Micro F1 score")
plt.title("Transitions of the micro F1 score")
plt.legend()
plt.savefig(figure_dir / "summary_of_micro_F1_score.png")

if spkind:
    plt.clf()
    for spk in range(spk_num):
        plt.plot(range(T), spkind_letter_ARIs[:, spk].mean(axis=0), label=f"{spk+1}-th speaker")
    plt.xlabel("Iteration")
    plt.ylabel("ARI")
    plt.title("Transitions of the letter ARI")
    plt.legend()
    plt.savefig(figure_dir / "speaker_individual_summary_of_letter_ARI.png")

    plt.clf()
    for spk in range(spk_num):
        plt.plot(range(T), spkind_word_ARIs[:, spk].mean(axis=0), label=f"{spk+1}-th speaker")
    plt.xlabel("Iteration")
    plt.ylabel("ARI")
    plt.title("Transitions of the word ARI")
    plt.legend()
    plt.savefig(figure_dir / "speaker_individual_summary_of_word_ARI.png")
print("Done!")

#%%
print("Save npy files...")
np.save(summary_dir / "resample_times.npy", resample_times)
np.save(summary_dir / "log_likelihoods.npy", log_likelihoods)

np.save(summary_dir / "letter_ARI.npy", letter_ARIs)
np.save(summary_dir / "letter_macro_F1.npy", letter_macro_f1_scores)
np.save(summary_dir / "letter_micro_F1.npy", letter_micro_f1_scores)
np.save(summary_dir / "word_ARI.npy", word_ARIs)
np.save(summary_dir / "word_macro_F1.npy", word_macro_f1_scores)
np.save(summary_dir / "word_micro_F1.npy", word_micro_f1_scores)

if spkind:
    np.save(summary_dir / "Speaker_individual_letter_ARI.npy", spkind_letter_ARIs)
    np.save(summary_dir / "Speaker_individual_word_ARI.npy", spkind_word_ARIs)
print("Done!")
