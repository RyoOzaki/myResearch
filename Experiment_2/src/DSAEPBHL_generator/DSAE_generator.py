import numpy as np


class DSAE_generator(object):

    def __init__(self, param_file, activator=np.tanh):
        self.activator = activator
        self.__set_params(param_file)

    def __set_params(self, param_file):
        param_npz = np.load(param_file)
        depth = int(len(param_npz.keys()) / 4)
        self.decoder_params = []
        for n in range(depth):
            weight = param_npz[f"{n+1}_th_network/decoder_weight"]
            bias = param_npz[f"{n+1}_th_network/decoder_bias"]
            self.decoder_params.append([weight, bias])

    def generate(self, input):
        out = input
        for weight, bias in self.decoder_params[::-1]:
            out = self.activator(np.dot(out, weight) + bias)
        return out
