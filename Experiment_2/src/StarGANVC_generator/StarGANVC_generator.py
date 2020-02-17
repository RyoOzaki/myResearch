import numpy as np
import chainer
from chainer import serializers
import chainer.functions as F
from .modules import stargan_net as net

generator_class = net.Generator1
class StarGANVC_generator(object):

    def __init__(self, generator_path, speaker_num):
        self.generator = generator_class(speaker_num)
        serializers.load_npz(generator_path, self.generator)
        identity = np.identity(speaker_num, dtype=np.float32)
        speaker_vectors = [chainer.Variable(identity[i]) for i in range(speaker_num)]
        self.speaker_vectors = speaker_vectors

    def generate(self, input, target, reinput=False):
        out = self.generator.reconstruct(chainer.Variable(input.T), self.speaker_vectors[target])
        if reinput:
            out = self.generator(out[:, 0], self.speaker_vectors[target])
        out = np.squeeze(out.data)
        return out.T
