import numpy as np

def load_NPBDAA_parameters(param_file):
    param_npz = np.load(param_file)
    bigram = param_npz["bigram/trans_matrix"]
    weights = param_npz["bigram/weights"]
    return weights, bigram

def counting(sentences, depth):
    cnt_matrix = np.zeros((depth, depth))
    for snt in sentences:
        for i in range(1, snt.shape[0]):
            if snt[i] == -1:
                continue
            cnt_matrix[snt[i-1], snt[i]] += 1
    return cnt_matrix

def merge_bigram(weights, bigram, sentences, depth, BOS_index, EOS_index):
    cnt_matrix = counting(sentences, depth)
    merged_bigram = np.zeros((depth, depth))

    depth_range = np.arange(depth)
    depth_flag = np.logical_and(depth_range != BOS_index, depth_range != EOS_index)
    # merged_bigram[BOS_index] = cnt_matrix[BOS_index] / cnt_matrix[BOS_index].sum()
    merged_bigram[BOS_index, depth_flag] = weights
    for wrd in range(depth):
        if wrd in [BOS_index, EOS_index]:
            continue
        wrd_eos = cnt_matrix[wrd, EOS_index] / cnt_matrix[wrd].sum()
        wrd_not_eos = 1 - wrd_eos

        merged_bigram[wrd, depth_flag] = bigram[wrd] * wrd_not_eos
        merged_bigram[wrd, EOS_index] = wrd_eos
    return merged_bigram

class Bigram_generator(object):

    def __init__(self, sentences_file, parameter_file):
        weights, bigram = load_NPBDAA_parameters(parameter_file)
        sentences_npz = np.load(sentences_file)
        sentences = sentences_npz["sentences"]
        depth = sentences_npz["depth"]
        BOS_index = sentences_npz["BOS"]
        EOS_index = sentences_npz["EOS"]
        merged_bigram = merge_bigram(weights, bigram, sentences, depth, BOS_index, EOS_index)
        # merged_bigram /= merged_bigram.sum(axis=1, keepdims=True)

        self.depth = depth
        self.bigram = merged_bigram
        self.BOS_index = BOS_index
        self.EOS_index = EOS_index

    def generate(self, size=1, unique=False):
        sentences = []
        if unique:
            i = 0
            while i < size:
                wrd = self.BOS_index
                snt = [wrd, ]
                while wrd != self.EOS_index:
                    wrd = np.random.choice(self.depth, p=self.bigram[wrd])
                    snt.append(wrd)
                if snt[1:-1] in sentences or len(snt) <= 2:
                    continue
                sentences.append(snt[1:-1])
                i += 1
        else:
            for s in range(size):
                wrd = self.BOS_index
                snt = [wrd, ]
                while wrd != self.EOS_index:
                    wrd = np.random.choice(self.depth, p=self.bigram[wrd])
                    snt.append(wrd)
                sentences.append(snt[1:-1])

        return sentences
