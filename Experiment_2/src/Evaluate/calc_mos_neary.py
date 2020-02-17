import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import editdistance # pip install editdistance
from itertools import product
import re
import matplotlib.pyplot as plt

# Case	 Target_speaker	 MOS	 Neary

# def trimming(s):
#     s = s.strip()
#     # print(f"==== in:{s}")
#     words = ["a", "i", "u", "e", "o", "n"]
#     for w in words:
#         s = re.sub(f"{w}+", w, s)
#     # print(f"====out:{s}")
#     return s


parser = ArgumentParser()

parser.add_argument("--score_files", type=Path, nargs="+")
parser.add_argument("--cases", type=str, nargs="+")
parser.add_argument("--speaker", type=str, required=True)
parser.add_argument("--num_of_test", type=int, required=True)
parser.add_argument("--load_npy", action="store_true")
# parser.add_argument("--sentences", type=str, nargs="+")

args = parser.parse_args()

cases = args.cases
# speakers = sorted(list(set(map(str, np.load(args.speaker_id).values()))))
speakers = [args.speaker, ]
# sentences = [snt.replace("_", "") for snt in args.sentences]

if not args.load_npy:
    case_N = len(cases)
    speaker_N = len(speakers)
    trans_N = len(args.score_files)
    snt_N = args.num_of_test

    neary_matrix = np.ones((trans_N, case_N, speaker_N, snt_N)) * -1
    mos_matrix = np.ones((trans_N, case_N, speaker_N, snt_N)) * -1

    counters = np.empty((case_N, speaker_N), dtype=int)

    for t, file in enumerate(args.score_files):
        print(f"{t+1}-th transcription: {file}")
        counters *= 0
        lines = file.read_text().split("\n")
        lines = [l for l in lines if l]
        lines = [list(map(lambda s: s.strip(), l.split(","))) for l in lines]

        case_idx = lines[0].index("Case")
        ts_idx = lines[0].index("Target_speaker")
        neary_idx = lines[0].index("Neary")
        mos_idx = lines[0].index("MOS")

        for i in range(1, len(lines)):
            case_i = cases.index(lines[i][case_idx])
            ts_i = speakers.index(lines[i][ts_idx])
            idx = counters[case_i, ts_i]
            mos = int(lines[i][mos_idx])
            neary = int(lines[i][neary_idx])

            neary_matrix[t, case_i, ts_i, idx] = neary
            mos_matrix[t, case_i, ts_i, idx] = mos
            counters[case_i, ts_i] += 1
    assert np.all(neary_matrix != -1)
    assert np.all(mos_matrix != -1)
    np.save("t2_NEARY.npy", neary_matrix)
    np.save("t2_MOS.npy", mos_matrix)

NEARY = np.load("t2_NEARY.npy")
MOS = np.load("t2_MOS.npy")

print()
print("NEARY")
for i, c in enumerate(cases):
    print(f"  {c}: {NEARY[:, i].mean():.3f} \pm {NEARY[:, i].std():.3f}")
    print(f"  {c}: var:{NEARY[:, i].var():.3f}, size:{np.product(NEARY[:, i].shape)}")

# print()
# print("CER Case-ident or not")
# identity = np.identity(speaker_N, dtype=bool)
# for (k, case) in enumerate(cases):
#     print(f"  {case}-ident: {CER[:, k, identity].mean():.3f} \pm {CER[:, k, identity].std():.3f}")
#     print(f"  {case}-not ident: {CER[:, k, ~identity].mean():.3f} \pm {CER[:, k, ~identity].std():.3f}")

print()
print("MOS")
for i, c in enumerate(cases):
    print(f"  {c}: {MOS[:, i].mean():.3f} \pm {MOS[:, i].std():.3f}")
    print(f"  {c}: var:{MOS[:, i].var():.3f}, size:{np.product(MOS[:, i].shape)}")

plt.rcParams["font.size"] = 16
#======= Plotingcase_1 - MOS
plt.clf()
score_matrix = MOS
score_name = "MOS"
ylim = (0, 5)
case_1 = "stargan_unigram"
case_2 = "stargan_bigram"
case_3 = "stargan_lstm"
xticks = ["Unigram", "Bigram", "LSTM-LM"]
i = cases.index(case_1)
j = cases.index(case_2)
k = cases.index(case_3)
dat_1 = score_matrix[:, i].reshape(-1)
dat_2 = score_matrix[:, j].reshape(-1)
dat_3 = score_matrix[:, k].reshape(-1)
x = np.arange(3)
dats = [dat_1, dat_2, dat_3]
means = [np.mean(d) for d in dats]
stds = [np.std(d) for d in dats]
colors = ["tab:blue", "tab:orange", "tab:green"]
plt.bar(x, means, capsize=5, yerr=stds, color=colors)
plt.xticks(x, xticks)
plt.ylabel(score_name)
plt.ylim(ylim)
# plt.show()
plt.savefig("test_2_mos.png")

#======= Plotingcase_1 - NEARY
plt.clf()
score_matrix = NEARY
score_name = "Similarity"
ylim = (0, 5)
case_1 = "stargan_unigram"
case_2 = "stargan_bigram"
case_3 = "stargan_lstm"
xticks = ["Unigram", "Bigram", "LSTM-LM"]
i = cases.index(case_1)
j = cases.index(case_2)
k = cases.index(case_3)
dat_1 = score_matrix[:, i].reshape(-1)
dat_2 = score_matrix[:, j].reshape(-1)
dat_3 = score_matrix[:, k].reshape(-1)
x = np.arange(3)
dats = [dat_1, dat_2, dat_3]
means = [np.mean(d) for d in dats]
stds = [np.std(d) for d in dats]
colors = ["tab:blue", "tab:orange", "tab:green"]
plt.bar(x, means, capsize=5, yerr=stds, color=colors)
plt.xticks(x, xticks)
plt.ylabel(score_name)
plt.ylim(ylim)
# plt.show()
plt.savefig("test_2_similarity.png")

# #======= T-test
# import scipy.stats as stats
# print()
# print("For t-test")
# case_1 = "stargan_unigram"
# case_2 = "stargan_bigram"
# i = cases.index(case_1)
# j = cases.index(case_2)
#
# score_matrix = NEARY
#
# # ttest_result = stats.ttest_ind(
# #     score_matrix[:, i].reshape(-1),
# #     score_matrix[:, j].reshape(-1)
# # )
# ttest_result = stats.ttest_rel(
#     score_matrix[:, i].reshape(-1),
#     score_matrix[:, j].reshape(-1)
# )
# print(ttest_result)
# print(f"p < 0.05 : {ttest_result.pvalue < 0.05}")
# print(f"p < 0.01 : {ttest_result.pvalue < 0.01}")

import scipy.stats as stats
from itertools import combinations
print()
print("For t-test")
test_cases = ["stargan_unigram", "stargan_bigram", "stargan_lstm"]
correction_coeff = len(test_cases) * (len(test_cases) - 1) / 2
for case_1, case_2 in combinations(test_cases, 2):
    # case_1 = "stargan_unigram"
    # case_2 = "stargan_bigram"
    i = cases.index(case_1)
    j = cases.index(case_2)

    score_matrix = NEARY

    # ttest_result = stats.ttest_ind(
    #     score_matrix[:, i].reshape(-1),
    #     score_matrix[:, j].reshape(-1)
    # )
    ttest_result = stats.ttest_rel(
        score_matrix[:, i].reshape(-1),
        score_matrix[:, j].reshape(-1)
    )
    # print(ttest_result)
    # print(f"p < 0.05 : {ttest_result.pvalue < 0.05}")
    # print(f"p < 0.01 : {ttest_result.pvalue < 0.01}")
    if ttest_result.pvalue < 0.01 / correction_coeff:
        print(f"{case_1}<->{case_2} : p < 0.01")
    elif ttest_result.pvalue < 0.05 / correction_coeff:
        print(f"{case_1}<->{case_2} : p < 0.05")
