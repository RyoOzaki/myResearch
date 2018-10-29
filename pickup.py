import numpy as np
from pathlib import Path
import shutil

labels = np.loadtxt("4_word_duplication_index.txt", dtype=str)

timit_root = Path("/home/ema/timit/TIMIT")
source = Path("/home/ema/work/timit_extract/source")

common_labels = ["SA1", "SA2"]
# common_labels = []

for label in labels:
    tmp = timit_root.glob("**/{}.WAV".format(label))
    speaker_dirs = [f.parent for f in tmp]
    for dir in speaker_dirs:
        target_dir = (source / dir.name)
        target_dir.mkdir(exist_ok=True)
        target_files = dir.glob("*{}*".format(label))
        for target in target_files:
            shutil.copy(str(target), str(target_dir / target.name))
        for clab in common_labels:
            target_files = dir.glob("*{}*".format(clab))
            for target in target_files:
                shutil.copy(str(target), str(target_dir / target.name))
