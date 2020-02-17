import numpy as np

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def calc_params(letter_num, label_src, mcep_src):
    phn_N = letter_num
    label_src = label_src[:mcep_src.shape[0]]

    mcep_dim = mcep_src.shape[1]
    mcep_means = []
    mcep_stds = []
    for phn in range(phn_N):
        mcep = mcep_src[label_src == phn]
        if mcep.shape[0] != 0:
            mcep_means.append(mcep.mean(axis=0))
            mcep_stds.append(mcep.std(axis=0))
        else:
            mcep_means.append(np.zeros(mcep_dim))
            mcep_stds.append(np.zeros(mcep_dim))
    return mcep_means, mcep_stds

class MCEP_generator(object):

    def __init__(self, letter_num, mcep, letter_stateseq=None, using_iter=-1, gold_transcription=False, mode="ML"):
        self.mode = mode
        concat_mcep, _ = packing(mcep)
        if gold_transcription:
            concat_label = np.concatenate(letter_stateseq, axis=0)
        else:
            concat_label, _ = packing([lseq[using_iter] for lseq in letter_stateseq])
        mcep_means, mcep_stds = calc_params(letter_num, concat_label, concat_mcep)
        self.mcep_means = np.array(mcep_means)
        self.mcep_stds = np.array(mcep_stds)
        self.mcep_dim = self.mcep_means.shape[1]

    def generate(self, letter_stateseq):
        if self.mode == "ML":
            return self.mcep_means[letter_stateseq]
        elif self.mode == "RND":
            gen_mcep = np.zeros((letter_stateseq.shape[0], self.mcep_dim))
            for phn, (mean, std) in enumerate(zip(self.mcep_means, self.mcep_stds)):
                flag = (letter_stateseq == phn)
                N = np.sum(flag)
                gen_mcep[flag] = np.random.randn(N, self.mcep_dim) * std + mean
            return gen_mcep
