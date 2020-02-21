import sys
import numpy as np
import matplotlib
if sys.platform in ['linux', 'linux2']:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import six
import os
import math, argparse, random
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
import itertools
from pathlib import Path
import modules.stargan_net as net
from util.utility import separate_speaker, get_separated_values
from tqdm import trange

# make ramdom indexes sequence (N kinds, length of list = Nmax)
def myperm(N, Nmax):
    rep = math.ceil(Nmax/N)
    indexes = np.concatenate([np.random.permutation(N) for _ in range(rep)])

    return indexes[:Nmax]

def packing(np_objs):
    lengths = [data.shape[0] for data in np_objs]
    return np.concatenate(np_objs, axis=0), lengths

def unpacking(np_obj, lengths):
    cumsum_lens = np.concatenate(([0], np.cumsum(lengths)))
    N = len(lengths)
    return [np_obj[cumsum_lens[i]:cumsum_lens[i+1]] for i in range(N)]

# input: list of bach datas [(mcep_dim, T1), (mcep_dim, T2), ... ]
# return: np.array which shape is (batch_size, mcep_dim, max(T1, T2, ... ))
# if mcep_dim is difference, I think return error.
def batchlist2array(batchlist):
    # batchlist[b]
    # b: utterance index
    batchsize = len(batchlist)
    widths = [batchdata.shape[1] for batchdata in batchlist]
    maxheight = batchlist[0].shape[0]
    maxwidth = max(widths)

    X = np.zeros((batchsize, maxheight, maxwidth))
    for b in range(batchsize):
        tmp = batchlist[b]
        tmp = np.tile(tmp, (1, math.ceil(maxwidth/tmp.shape[1])))
        X[b,:,:] = tmp[:, 0:maxwidth] # error if mcep_dim is different
        #X[b,0:tmp.shape[0],0:tmp.shape[1]] = tmp
        #mask[b,:,0:tmp.shape[1]] = 1.0
    return X

def snapshot(output_dir, epoch, generator, classifier, adverserial_discriminator):
    # print('save the generator at {} epoch'.format(epoch))
    serializers.save_npz(output_dir / f'{epoch}.gen', generator)
    # print('save the classifier at {} epoch'.format(epoch))
    serializers.save_npz(output_dir / f'{epoch}.cls', classifier)
    # print('save the real/fake discriminator at {} epoch'.format(epoch))
    serializers.save_npz(output_dir / f'{epoch}.advdis', adverserial_discriminator)

# print('AdvLoss_d={}, AdvLoss_g={}, ClsLoss_r={}, ClsLoss_f={}'
#       .format(AdvLoss_d.data, AdvLoss_g.data, ClsLoss_r.data, ClsLoss_f.data))
# print('CycLoss={}, RecLoss={}'
#       .format(CycLoss.data, RecLoss.data))
def save_loss(output_dir, advloss_d, advloss_g, clsloss_r, clsloss_f, cycloss, recloss):
    logdir = output_dir / "sgvc_log"
    logdir.mkdir(exist_ok=True)
    fnames = ["advloss_d", "advloss_g", "clsloss_r", "clsloss_f", "cycloss", "recloss"]
    values = chainer.cuda.to_cpu([advloss_d, advloss_g, clsloss_r, clsloss_f, cycloss, recloss])
    for fname, value in zip(fnames, values):
        with (logdir / f"{fname}.txt").open(mode="a") as f:
            np.savetxt(f, np.array([value, ]))

def main():
    parser = argparse.ArgumentParser(description='Train stargan voice convertor')
    parser.add_argument(
        '--gpu', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument("--train_data", type=Path, required=True, help="training data")
    parser.add_argument("--speaker_id", type=Path, required=True, help="speaker_id file")
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument(
        '--epoch', default=6000, type=int, help='number of epochs to learn')
    parser.add_argument("--epoch_start", type=int, default=0)

    parser.add_argument(
        '--snapshot', default=100, type=int, help='interval of snapshot')
    parser.add_argument(
        '--batchsize', type=int, default=4, help='Batch size')
    parser.add_argument(
        '--optimizer', default='Adam', choices=["Adam", "MomentumSGD", "RMSprop"], type=str, help='optimizer to use: Adam, MomentumSGD, RMSprop')
    parser.add_argument(
        '--lrate', default='0.00001', type=float, help='learning rate for Adam, MomentumSGD or RMSprop')
    parser.add_argument(
        '--genpath', type=str, help='path for a pretrained generator')
    parser.add_argument(
        '--clspath', type=str, help='path for a pretrained classifier')
    parser.add_argument(
        '--advdispath', type=str, help='path for a pretrained real/fake discriminator')

    args = parser.parse_args()
    epsi = sys.float_info.epsilon

    output_file = args.output_file
    output_dir = output_file.with_suffix("")
    output_dir.mkdir(exist_ok=True, parents=True)

    all_source = np.load(args.train_data)
    Speakers, SpeakerIndividualKeys = separate_speaker(np.load(args.speaker_id))
    NormalizedAllData = get_separated_values(all_source, SpeakerIndividualKeys)
    SpeakerNum = len(Speakers)

    # Set input directories
    EpochNum = args.epoch
    BatchSize = args.batchsize

    SentenceNum = [len(SpeakerIndividualKeys[s]) for s in range(SpeakerNum)]
    MaxSentenceNum = max(SentenceNum)

    print('#GPU: {}'.format(args.gpu))
    print('#epoch: {}'.format(EpochNum))
    print('Optimizer: {}'.format(args.optimizer))
    print('Learning rate: {}'.format(args.lrate))
    print('Snapshot: {}'.format(args.snapshot))

    # Set up model
    num_mels = 36
    zdim = 5
    hdim = 32
    cdim = 8
    adim = 32

    # num_mels = data.shape[0] (36dim)
    # zdim = 8
    # hdim = 32
    generator_class = net.Generator_new
    classifier_class = net.Classifier1
    discriminator_class = net.AdvDiscriminator1
    loss_class = net.Loss_new

    generator = generator_class(SpeakerNum)
    paranum = sum(p.data.size for p in generator.params())
    print('Parameter #: {}'.format(paranum))

    # cdim = 8
    classifier = classifier_class(num_mels, SpeakerNum, cdim)
    paranum = sum(p.data.size for p in classifier.params())
    print('Parameter #: {}'.format(paranum))

    # adim = 32
    adverserial_discriminator = discriminator_class(num_mels, SpeakerNum, adim)
    # adverserial_discriminator = net.AdvDiscriminator_noactive(num_mels, SpeakerNum, adim)
    paranum = sum(p.data.size for p in adverserial_discriminator.params())
    print('Parameter #: {}'.format(paranum))

    if args.genpath is not None:
        try:
            serializers.load_npz(args.genpath, generator)
        except:
            print('Could not load generator.')
    if args.clspath is not None:
        try:
            serializers.load_npz(args.clspath, classifier)
        except:
            print('Could not load domain classifier.')
    if args.advdispath is not None:
        try:
            serializers.load_npz(args.advdispath, adverserial_discriminator)
        except:
            print('Could not load real/fake discriminator.')

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        generator.to_gpu()
        classifier.to_gpu()
        adverserial_discriminator.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy

    # Set up optimziers
    # loss = net.Loss1(generator, classifier, adverserial_discriminator)
    loss = loss_class(generator, classifier, adverserial_discriminator)
    # w_adv = 1.0
    # w_cls = 1.0
    # w_cyc = 1.0
    # w_rec = 1.0
    w_adv = 1.0
    w_cls = 1.0
    w_cyc = 1.0
    w_rec = 1.0
    if args.optimizer == 'MomentumSGD':
        opt_gen = optimizers.MomentumSGD(lr=args.lrate, momentum=0.9)
        opt_cls = optimizers.MomentumSGD(lr=args.lrate, momentum=0.9)
        opt_advdis = optimizers.MomentumSGD(lr=args.lrate, momentum=0.9)
    elif args.optimizer == 'Adam':
        opt_gen = optimizers.Adam(alpha=0.001, beta1=0.9)
        opt_cls = optimizers.Adam(alpha=0.00005, beta1=0.5)
        opt_advdis = optimizers.Adam(alpha=0.00001, beta1=0.5)
    elif args.optimizer == 'RMSprop':
        opt_gen = optimizers.RMSprop(lr=args.lrate)
        opt_cls = optimizers.RMSprop(lr=args.lrate)
        opt_advdis = optimizers.RMSprop(lr=args.lrate)
    opt_gen.setup(generator)
    opt_cls.setup(classifier)
    opt_advdis.setup(adverserial_discriminator)


    AllCombinationPairs = list(itertools.combinations(range(SpeakerNum), 2))
    # train
    for epoch in trange(args.epoch_start, EpochNum+1):

        # shuffled_indexes[speaker_idx][idx]: value is index of NormalizedAllData[speaker_idx][**here**]
        shuffled_indexes = [myperm(SentenceNum[s], MaxSentenceNum) for s in range(SpeakerNum)]

        for n in range(MaxSentenceNum//BatchSize):
            # batchlist_mcep[speaker_idx][sentence_idx_in_batch]
            batchlist_mcep = []
            begin_idx = n * BatchSize
            end_idx = begin_idx + BatchSize # not include @ end_idx
            for s in range(SpeakerNum):
                batch_tmp = []
                for idx in shuffled_indexes[s][begin_idx:end_idx]:
                    batch_tmp.append( NormalizedAllData[s][idx].T ) # Transpose here!!
                batchlist_mcep.append(batch_tmp)
            # Convert batchlist into a list of arrays
            X = [batchlist2array(batchlist) for batchlist in batchlist_mcep]

            xin = [chainer.Variable(xp.asarray(Xs, dtype=np.float32)) for Xs in X]

            # Iterate through all speaker pairs
            random.shuffle(AllCombinationPairs)
            for s0, s1 in AllCombinationPairs:
                AdvLoss_d, AdvLoss_g, ClsLoss_r, ClsLoss_f, CycLoss, RecLoss \
                    = loss.calc_loss(xin[s0], xin[s1], s0, s1, SpeakerNum)
                gen_loss = (w_adv * AdvLoss_g + w_cls * ClsLoss_f
                            + w_cyc * CycLoss + w_rec * RecLoss)
                cls_loss = ClsLoss_r
                advdis_loss = AdvLoss_d
                generator.cleargrads()
                gen_loss.backward()
                opt_gen.update()
                classifier.cleargrads()
                cls_loss.backward()
                opt_cls.update()
                adverserial_discriminator.cleargrads()
                advdis_loss.backward()
                opt_advdis.update()

            print('epoch {}, mini-batch {}:'.format(epoch, n+1))
            print('AdvLoss_d={}, AdvLoss_g={}, ClsLoss_r={}, ClsLoss_f={}'
                  .format(AdvLoss_d.data, AdvLoss_g.data, ClsLoss_r.data, ClsLoss_f.data))
            print('CycLoss={}, RecLoss={}'
                  .format(CycLoss.data, RecLoss.data))
        save_loss(output_dir, AdvLoss_d.data, AdvLoss_g.data, ClsLoss_r.data, ClsLoss_f.data, CycLoss.data, RecLoss.data)

        if epoch % args.snapshot == 0:
            snapshot_dir = output_dir / "snapshot"
            snapshot_dir.mkdir(exist_ok=True)
            snapshot(snapshot_dir, epoch, generator, classifier, adverserial_discriminator)
            snapshot_feature_dir = output_dir / "snapshot_feature"
            snapshot_feature_dir.mkdir(exist_ok=True)
            output = {}
            with chainer.no_backprop_mode():
                identity = np.identity(SpeakerNum)
                for s in range(SpeakerNum):
                    speaker_vec = chainer.Variable(xp.asarray(identity[s], dtype=np.float32))
                    for key, mcep in zip(SpeakerIndividualKeys[s], NormalizedAllData[s]):
                        mcep_T = mcep.T
                        out = generator.hidden_layer(
                            chainer.Variable(xp.asarray(mcep_T[np.newaxis,:,:], dtype=np.float32)),
                            speaker_vec
                            )
                        out = np.squeeze(cuda.to_cpu(out.data))
                        output[key] = out.T
            np.savez(snapshot_feature_dir / f"{output_file.stem}_epoch_{epoch:05}.npz", **output)

    # output final result
    output = {}
    with chainer.no_backprop_mode():
        identity = np.identity(SpeakerNum)
        for s in range(SpeakerNum):
            speaker_vec = chainer.Variable(xp.asarray(identity[s], dtype=np.float32))
            for key, mcep in zip(SpeakerIndividualKeys[s], NormalizedAllData[s]):
                mcep_T = mcep.T
                out = generator.hidden_layer(
                    chainer.Variable(xp.asarray(mcep_T[np.newaxis,:,:], dtype=np.float32)),
                    speaker_vec
                    )
                out = np.squeeze(cuda.to_cpu(out.data))
                output[key] = out.T
    np.savez(output_file, **output)

if __name__ == '__main__':
    main()
