import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import editdistance # pip install editdistance
from itertools import product
import re

# Case	 Source_speaker	 Target_speaker	 Grand_truth	 Transcription

def trimming(s):
    s = s.strip()
    # print(f"==== in:{s}")
    words = ["a", "i", "u", "e", "o", "n"]
    for w in words:
        s = re.sub(f"{w}+", w, s)
    # print(f"====out:{s}")
    return s
parser = ArgumentParser()

parser.add_argument("--transcriptions", action="append", type=Path, nargs="+")
parser.add_argument("--cases", type=str, nargs="+")
parser.add_argument("--speaker_id", type=Path, required=True)
parser.add_argument("--sentences", type=str, nargs="+")
parser.add_argument("--load_npy", action="store_true")

args = parser.parse_args()

cases = args.cases
speakers = sorted(list(set(map(str, np.load(args.speaker_id).values()))))
sentences = [snt.replace("_", "") for snt in args.sentences]

if not args.load_npy:

    case_N = len(cases)
    speaker_N = len(speakers)
    trans_N = len(args.transcriptions)
    snt_N = len(sentences)

    cer_matrix = np.ones((trans_N, case_N, speaker_N, speaker_N, snt_N)) * -1
    mos_matrix = np.ones((trans_N, case_N, speaker_N, speaker_N, snt_N)) * -1

    for t, transfiles in enumerate(args.transcriptions):
        print(f"{t+1}-th transcription: {transfiles}")
        for f in transfiles:
            lines = f.read_text().split("\n")
            lines = [l for l in lines if l]
            lines = [list(map(lambda s: s.strip(), l.split(","))) for l in lines]

            case_idx = lines[0].index("Case")
            ss_idx = lines[0].index("Source_speaker")
            ts_idx = lines[0].index("Target_speaker")
            truth_idx = lines[0].index("Grand_truth")
            trans_idx = lines[0].index("Transcription")
            mos_idx = lines[0].index("MOS")

            for i in range(1, len(lines)):
                case_i = cases.index(lines[i][case_idx])
                ss_i = speakers.index(lines[i][ss_idx])
                ts_i = speakers.index(lines[i][ts_idx])
                truth = lines[i][truth_idx]
                truth_i = sentences.index(truth)
                trans = trimming(lines[i][trans_idx])
                mos = int(lines[i][mos_idx])

                cer = editdistance.eval(truth, trans) / len(truth)
                cer_matrix[t, case_i, ss_i, ts_i, truth_i] = cer
                mos_matrix[t, case_i, ss_i, ts_i, truth_i] = mos
    assert np.all(cer_matrix != -1)
    assert np.all(mos_matrix != -1)
    np.save("t1_CER.npy", cer_matrix)
    np.save("t1_MOS.npy", mos_matrix)

CER = np.load("t1_CER.npy")
MOS = np.load("t1_MOS.npy")

print()
print("CER")
for i, c in enumerate(cases):
    print(f"  {c}: {CER[:, i].mean():.3f} \pm {CER[:, i].std():.3f}")
    print(f"  {c}: var:{CER[:, i].var():.3f}, size:{np.product(CER[:, i].shape)}")

print()
print("MOS")
for i, c in enumerate(cases):
    print(f"  {c}: {MOS[:, i].mean():.3f} \pm {MOS[:, i].std():.3f}")
    print(f"  {c}: var:{MOS[:, i].var():.3f}, size:{np.product(MOS[:, i].shape)}")


import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 16
#======= Plotingcase_1 - MOS
plt.clf()
score_matrix = MOS
score_name = "MOS"
ylim = (0, 5)
case_1 = "topline"
case_2 = "baseline"
case_3 = "dsae_pbhl"
case_4 = "stargan"
xticks = ["topline", "baseline", "dsae_pbhl", "stargan"]
i = cases.index(case_1)
j = cases.index(case_2)
k = cases.index(case_3)
l = cases.index(case_4)
dat_1 = score_matrix[:, i].reshape(-1)
dat_2 = score_matrix[:, j].reshape(-1)
dat_3 = score_matrix[:, k].reshape(-1)
dat_4 = score_matrix[:, l].reshape(-1)
x = np.arange(4)
dats = [dat_1, dat_2, dat_3, dat_4]
means = [np.mean(d) for d in dats]
stds = [np.std(d) for d in dats]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
plt.bar(x, means, capsize=5, yerr=stds, color=colors)
plt.xticks(x, xticks)
plt.ylabel(score_name)
plt.ylim(ylim)
# plt.show()
plt.savefig("test_1_mos.png")



#======= Plotingcase_1 - CER
plt.clf()
score_matrix = CER
score_name = "CER"
ylim = (0, 1)
case_1 = "topline"
case_2 = "baseline"
case_3 = "dsae_pbhl"
case_4 = "stargan"
xticks = ["topline", "baseline", "dsae_pbhl", "stargan"]
i = cases.index(case_1)
j = cases.index(case_2)
k = cases.index(case_3)
l = cases.index(case_4)
dat_1 = score_matrix[:, i].reshape(-1)
dat_2 = score_matrix[:, j].reshape(-1)
dat_3 = score_matrix[:, k].reshape(-1)
dat_4 = score_matrix[:, l].reshape(-1)
x = np.arange(4)
dats = [dat_1, dat_2, dat_3, dat_4]
means = [np.mean(d) for d in dats]
stds = [np.std(d) for d in dats]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
plt.bar(x, means, capsize=5, yerr=stds, color=colors)
plt.xticks(x, xticks)
plt.ylabel(score_name)
plt.ylim(ylim)
# plt.show()
plt.savefig("test_1_cer.png")


import scipy.stats as stats
from itertools import combinations
print()
print("For t-test")
test_cases = ["topline", "baseline", "dsae_pbhl", "stargan"]
correction_coeff = len(test_cases) * (len(test_cases) - 1) / 2
for case_1, case_2 in combinations(test_cases, 2):
    # case_1 = "stargan"
    # case_2 = "topline"
    i = cases.index(case_1)
    j = cases.index(case_2)

    score_matrix = MOS

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
