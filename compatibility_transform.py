import numpy as np

data = np.load("feature/concat_data.npz")
phoneme = np.load("feature/phoneme_label.npz")
word = np.load("feature/word_label.npz")

for key in data.keys():
    np.savetxt("DATA/" + key + ".txt", data[key])
    np.savetxt("LABEL/" + key + ".lab", phoneme[key])
    np.savetxt("LABEL/" + key + ".lab2", word[key])
