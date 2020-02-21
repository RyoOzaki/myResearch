import numpy as np
import matplotlib
import sys
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import math, random
from scipy import signal
from scipy.io import wavfile
from argparse import ArgumentParser
import itertools
from pathlib import Path
import modules.stargan_net as net
from util.utility import separate_speaker, get_separated_values
import pyworld as pw
import chainer
from chainer import serializers

def denorm_mcep(generated_mcep, logf0, domain_mean, domain_std):
    gmcep_mean = generated_mcep[logf0!=0].mean(axis=0)
    gmcep_std = generated_mcep[logf0!=0].std(axis=0)

    domain_mcep = domain_std / gmcep_std * (generated_mcep - gmcep_mean) + domain_mean
    return domain_mcep

def conv_f0(f0, log_domain_mean, log_domain_std):
    lf0 = np.zeros_like(f0)
    lf0[f0!=0] = np.log(f0[f0!=0])
    lf0_mean = lf0[f0!=0].mean()
    lf0_std = lf0[f0!=0].std()

    cf0 = np.zeros_like(f0)
    cf0[f0!=0] = np.exp(log_domain_std / lf0_std * (lf0[f0!=0] - lf0_mean) + log_domain_mean)
    return cf0

generator_class = net.Generator1
discriminator_class = net.AdvDiscriminator1
classifier_class = net.Classifier1

gen_suffix = ".gen.npz"
dis_suffix = ".advdis.npz"
cls_suffix = ".cls.npz"
parser = ArgumentParser()

parser.add_argument("--snapshot_dir", type=Path)
parser.add_argument("--snapshot_name", type=str)

parser.add_argument('--generator', type=Path, help='path for a pretrained generator')
parser.add_argument('--discriminator', type=Path, help='path for a pretrained real/fake discriminator')
parser.add_argument('--classifier', type=Path, help='path for a pretrained classifier')

parser.add_argument("--speaker_id", type=Path, required=True)
parser.add_argument("--mcep", type=Path, required=True)
parser.add_argument("--f0", type=Path, required=True)
parser.add_argument("--ap", type=Path, required=True)

parser.add_argument("--samplerate", type=int, default=48000)
parser.add_argument("--fftsize", type=int, default=1024)
parser.add_argument("--frame_period", type=float, default=5E-3)

parser.add_argument("--mcep_norm_param", type=Path, nargs=2)
parser.add_argument("--logf0_norm_param", type=Path, nargs=2)

parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--flatten_dir", action="store_true")

args = parser.parse_args()

gen_path = args.generator or (args.snapshot_dir / args.snapshot_name).with_suffix(gen_suffix)
dis_path = args.discriminator or (args.snapshot_dir / args.snapshot_name).with_suffix(dis_suffix)
cls_path = args.classifier or (args.snapshot_dir / args.snapshot_name).with_suffix(cls_suffix)

# Set up model
num_mels = 36
zdim = 5
hdim = 32
cdim = 8
adim = 32

speakers, speaker_individual_keys = separate_speaker(np.load(args.speaker_id))
speaker_num = len(speakers)
identity = np.identity(speaker_num, dtype=np.float32)

spkind_mcep = get_separated_values(np.load(args.mcep), speaker_individual_keys)
spkind_f0 = get_separated_values(np.load(args.f0), speaker_individual_keys)
spkind_ap = get_separated_values(np.load(args.ap), speaker_individual_keys)

mcep_mean = np.load(args.mcep_norm_param[0])
mcep_std = np.load(args.mcep_norm_param[1])
logf0_mean = np.load(args.logf0_norm_param[0])
logf0_std = np.load(args.logf0_norm_param[1])

generator = generator_class(speaker_num)
adverserial_discriminator = discriminator_class(num_mels, speaker_num, adim)

serializers.load_npz(gen_path, generator)
serializers.load_npz(dis_path, adverserial_discriminator)

spkind_kmfa = [speaker_individual_keys, spkind_mcep, spkind_f0, spkind_ap]

with chainer.no_backprop_mode():
    # for tspk_idx, (fspk, tspk) in enumerate(speakers):
    real_flags = []
    for fspk_idx, tspk_idx in itertools.product(range(speaker_num), repeat=2):
        fspk_kmfa = [kmfa[fspk_idx] for kmfa in spkind_kmfa]
        fspk = speakers[fspk_idx]
        tspk = speakers[tspk_idx]
        print(f"Convert {fspk} -> {tspk}")
        fspk_lab = identity[fspk_idx]
        tspk_lab = identity[tspk_idx]
        tspk_dir = args.output_dir / f"To_{tspk}"
        tspk_dir.mkdir(exist_ok=True, parents=True)
        i = 0
        fake_datas = {}
        real_datas = {}
        for key, mcep, f0, ap in zip(*fspk_kmfa):
            if  args.flatten_dir:
                out_name = key.replace("/", "_")
            else:
                out_name = key
            out_wavfile = tspk_dir / f"{out_name}.wav"
            out_wavfile.parent.mkdir(exist_ok=True, parents=True)

            # test generator
            mcep_T = np.asarray(mcep.T, dtype=np.float32)
            mcep_T = mcep_T.reshape((1, *mcep_T.shape))
            gen_mcep_var = generator(mcep_T, tspk_lab)
            gen_mcep = gen_mcep_var[0].data.T
            denorm_gen_mcep = denorm_mcep(gen_mcep, f0, mcep_mean[tspk], mcep_std[tspk])
            denorm_gen_mcep = signal.medfilt(denorm_gen_mcep, (5, 1))
            conved_f0 = conv_f0(f0, logf0_mean[tspk], logf0_std[tspk])

            specenv = pw.decode_spectral_envelope(denorm_gen_mcep, args.samplerate, args.fftsize)
            x = pw.synthesize(conved_f0, specenv, ap, args.samplerate, frame_period=args.frame_period*1000)
            x = x / max(abs(x)) * 30000
            x = x.astype(np.int16)

            wavfile.write(out_wavfile, args.samplerate, x)

            # test discriminator
            # test real data
            if fspk not in real_flags:
                real_datas[key] = np.squeeze(adverserial_discriminator(mcep_T, fspk_lab, dp_ratio=0.0)[1].data)
            # test fake data
            fake_datas[key] = np.squeeze(adverserial_discriminator(gen_mcep_var, tspk_lab, dp_ratio=0.0)[1].data)

        # save values of discriminator of fake data (fspk -> tspk)
        plt.clf()
        fake_values = np.concatenate([fake for fake in fake_datas.values()], axis=0)
        length = fake_values.shape[0]
        plt.bar(np.arange(length), height=fake_values)
        plt.savefig(args.output_dir / f"fake_from_{fspk}_to_{tspk}.png")

        if fspk not in real_flags:
            real_flags.append(fspk)
            plt.clf()
            real_values = np.concatenate([real for real in real_datas.values()], axis=0)
            length = real_values.shape[0]
            plt.bar(np.arange(length), height=real_values)
            plt.savefig(args.output_dir / f"real_{fspk}.png")
