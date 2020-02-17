import numpy as np

class DSAEPBHL_generator(object):

    def __init__(self, param_file, pb_param_file, activator=np.tanh):
        self.activator = activator
        self.__set_params(param_file, pb_param_file)

    def __set_params(self, param_file, pb_param_file):
        param_npz = np.load(param_file)
        pb_param_npz = np.load(pb_param_file)
        self.pb_means = dict(pb_param_npz)
        self.speakers = list(pb_param_npz.keys())
        self.speakers_num = len(self.speakers)
        depth = int(len(param_npz.keys()) / 4)
        self.decoder_params = []
        for n in range(depth):
            weight = param_npz[f"{n+1}_th_network/decoder_weight"]
            bias = param_npz[f"{n+1}_th_network/decoder_bias"]
            self.decoder_params.append([weight, bias])

        self.pb_hidden_dim = self.pb_means[self.speakers[0]].shape[0]
        self.feat_dim = self.decoder_params[-2][0].shape[0]

    def generate(self, input, target):
        pb_feat = np.ones((input.shape[0], self.pb_hidden_dim)) * self.pb_means[target]
        out = np.concatenate((input, pb_feat), axis=1)
        weight, bias = self.decoder_params[-1]
        out = self.activator(np.dot(out, weight) + bias)
        out = out[:, :self.feat_dim]
        for weight, bias in self.decoder_params[:-1][::-1]:
            out = self.activator(np.dot(out, weight) + bias)
        return out
