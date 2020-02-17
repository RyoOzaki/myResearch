import numpy as np
import matplotlib.pyplot as plt

src = np.load("feature/mcep_all_speaker_20msec.npz")

x = src["speaker_H/aioi_uo_ie"].T
y = src["speaker_K/aioi_uo_ie"].T

min_v = min([x.min(), y.min()])
offset = 2

plt.pcolormesh(x, vmin=min_v-offset)
plt.show()

plt.pcolormesh(y, vmin=min_v-offset)
plt.show()
