import numpy as np
from LSTM_language_model.model import LSTM_language_model

class LSTMLM_generator(object):

    def __init__(self, model_path, sentences_path):
        sentences_npz = np.load(sentences_path)
        depth = sentences_npz["depth"]
        BOS_index = sentences_npz["BOS"]
        EOS_index = sentences_npz["EOS"]
        self.LSTM_LM = LSTM_language_model(depth, None, BOS_index=BOS_index, EOS_index=EOS_index, load_model_path=model_path)

    def generate(self, size=1, unique=False):
        if unique:
            sentences = []
            i = 0
            while i < size:
                snt = self.LSTM_LM.generate()[0]
                if snt in sentences or len(snt) < 2:
                    continue
                sentences.append(snt)
                i += 1
        else:
            sentences = self.LSTM_LM.generate(size=size)
        for s in range(size):
            sentences[s] = sentences[s][1:-1]
        return sentences
