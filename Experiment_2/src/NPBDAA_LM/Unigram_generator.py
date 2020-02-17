import numpy as np

def make_unigram(sentences, depth):
    cnt_matrix = np.zeros((depth, ))
    words = sentences[sentences != -1]
    for w in words:
        cnt_matrix[int(w)] += 1
    cnt_matrix /= cnt_matrix.sum()
    return cnt_matrix

class Unigram_generator(object):

    def __init__(self, sentences_file):
        sentences_npz = np.load(sentences_file)
        sentences = sentences_npz["sentences"]
        depth = sentences_npz["depth"]
        BOS_index = sentences_npz["BOS"]
        EOS_index = sentences_npz["EOS"]
        unigram = make_unigram(sentences, depth)
        # merged_bigram /= merged_bigram.sum(axis=1, keepdims=True)

        self.depth = depth
        self.unigram = unigram
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
                    wrd = np.random.choice(self.depth, p=self.unigram)
                    if wrd == self.BOS_index:
                        continue
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
                    wrd = np.random.choice(self.depth, p=self.unigram)
                    snt.append(wrd)

                sentences.append(snt[1:-1])

        return sentences
