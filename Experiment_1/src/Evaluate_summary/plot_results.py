import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import japanize_matplotlib

figsize = (10, 8)

result_dir = Path("segmentation_result_summary")

word_ARI = np.load(result_di / "word_ARI.npz")
