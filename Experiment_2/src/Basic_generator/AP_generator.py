import numpy as np

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def expanding_label(label_src, width=4):
    ones = np.ones((label_src.shape[0], width))
    labs = ones * label_src.reshape((-1, 1))
    return labs.reshape((-1, ))

def calc_params(letter_num, label_src, ap_src):
    phn_N = letter_num
    label_src = expanding_label(label_src)
    label_src = label_src[:ap_src.shape[0]]

    ap_dim = ap_src.shape[1]
    ap_means = []
    ap_stds = []
    for phn in range(phn_N):
        ap = ap_src[label_src == phn]
        if ap.shape[0] == 0:
            ap_means.append(ap.mean(axis=0))
            ap_stds.append(ap.std(axis=0))
        else:
            ap_means.append(np.zeros(ap_dim))
            ap_stds.append(np.zeros(ap_dim))
    return ap_means, ap_stds

class AP_generator(object):

    def __init__(self, letter_num, ap, letter_stateseq=None, flat=False, using_iter=-1, mode="ML"):
        self.mode = mode
        concat_ap, _ = packing(ap)
        if flat:
            ap_means = [concat_ap.mean(axis=0) for _ in range(letter_num)]
            ap_stds = [concat_ap.std(axis=0) for _ in range(letter_num)]
        else:
            concat_label, _ = packing([lseq[using_iter] for lseq in letter_stateseq])
            ap_means, ap_stds = calc_params(letter_num, concat_label, concat_ap)
        self.ap_means = np.array(ap_means)
        self.ap_stds = np.array(ap_stds)
        self.ap_dim = self.ap_means.shape[1]

    def generate(self, letter_stateseq):
        if self.mode == "ML":
            return self.ap_means[letter_stateseq]
        elif self.mode == "RND":
            gen_ap = np.zeros((letter_stateseq.shape[0], self.ap_dim))
            for phn, (mean, std) in enumerate(zip(self.ap_means, self.ap_stds)):
                flag = (letter_stateseq == phn)
                N = np.sum(flag)
                gen_ap[flag] = np.random.randn(N, self.ap_dim) * std + mean
            return gen_ap
