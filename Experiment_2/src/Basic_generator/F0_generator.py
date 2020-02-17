import numpy as np

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def expanding_label(label_src, width=4):
    ones = np.ones((label_src.shape[0], width))
    labs = ones * label_src.reshape((-1, 1))
    return labs.reshape((-1, ))

def calc_params(letter_num, label_src, f0_src):
    phn_N = letter_num
    label_src = expanding_label(label_src)
    label_src = label_src[:f0_src.shape[0]]

    f0_means = []
    f0_stds = []
    for phn in range(phn_N):
        f0 = f0_src[label_src == phn]
        if f0.shape[0] != 0:
            f0_means.append(f0.mean(axis=0))
            f0_stds.append(f0.std(axis=0))
        else:
            f0_means.append(0)
            f0_stds.append(0)
    return f0_means, f0_stds

class F0_generator(object):

    def __init__(self, letter_num, f0, letter_stateseq=None, flat=False, using_iter=-1, mode="ML"):
        self.mode = mode
        self.letter_num = letter_num
        concat_f0, _ = packing(f0)
        if flat:
            f0_means = [concat_f0.mean(axis=0) for _ in range(letter_num)]
            f0_stds = [concat_f0.std(axis=0) for _ in range(letter_num)]
        else:
            concat_label, _ = packing([lseq[using_iter] for lseq in letter_stateseq])
            f0_means, f0_stds = calc_params(letter_num, concat_label, concat_f0)
        self.f0_means = np.array(f0_means)
        self.f0_stds = np.array(f0_stds)

    def generate(self, letter_stateseq):
        if self.mode == "ML":
            return self.f0_means[letter_stateseq]
        elif self.mode == "RND":
            gen_f0 = np.zeros(letter_stateseq.shape[0])
            for phn, (mean, std) in enumerate(zip(self.f0_means, self.f0_stds)):
                flag = (letter_stateseq == phn)
                N = np.sum(flag)
                gen_f0[flag] = np.random.randn(N) * std + mean
            return gen_f0
