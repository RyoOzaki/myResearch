import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--result_dir", type=Path, required=True, help="root directory of results")
parser.add_argument("--output_dir", type=Path, default="./summary")

args = parser.parse_args()

result_dir = args.result_dir
output_dir = args.output_dir
output_dir.mkdir(exist_ok=True, parents=True)

# /home/ozaki/Experiment_1/segmentation_result/dsae_all_speaker/summary_files
# letter_ARI.npy       letter_micro_F1.npy  resample_times.npy  word_macro_F1.npy
# letter_macro_F1.npy  log_likelihoods.npy  word_ARI.npy        word_micro_F1.npy
letter_ARI = {}
letter_micro_F1 = {}
letter_macro_F1 = {}
word_ARI = {}
word_micro_F1 = {}
word_macro_F1 = {}
resample_times = {}
log_likelihoods = {}

i = 0
for dir in result_dir.glob("*"):
    summary_dir = dir / "summary_files"
    if summary_dir.exists():
        letter_ARI[dir.stem] = np.load(summary_dir / "letter_ARI.npy")
        letter_micro_F1[dir.stem] = np.load(summary_dir / "letter_micro_F1.npy")
        letter_macro_F1[dir.stem] = np.load(summary_dir / "letter_macro_F1.npy")
        word_ARI[dir.stem] = np.load(summary_dir / "word_ARI.npy")
        word_micro_F1[dir.stem] = np.load(summary_dir / "word_micro_F1.npy")
        word_macro_F1[dir.stem] = np.load(summary_dir / "word_macro_F1.npy")
        resample_times[dir.stem] = np.load(summary_dir / "resample_times.npy")
        log_likelihoods[dir.stem] = np.load(summary_dir / "log_likelihoods.npy")

        i += 1

print(f"{i} results are collected!")

np.savez(output_dir / "letter_ARI.npz", **letter_ARI)
np.savez(output_dir / "letter_micro_F1.npz", **letter_micro_F1)
np.savez(output_dir / "letter_macro_F1.npz", **letter_macro_F1)
np.savez(output_dir / "word_ARI.npz", **word_ARI)
np.savez(output_dir / "word_micro_F1.npz", **word_micro_F1)
np.savez(output_dir / "word_macro_F1.npz", **word_macro_F1)
np.savez(output_dir / "resample_times.npz", **resample_times)
np.savez(output_dir / "log_likelihoods.npz", **log_likelihoods)

print(f"saved results npz to {output_dir}")
