import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()

parser.add_argument("--logdir", type=Path, required=True)
parser.add_argument("--dot", type=int, default=-1)
parser.add_argument("--begin_iter", type=int, default=0)

args = parser.parse_args()

log_dir = args.logdir
loss_file_names = ["advloss_d.txt",  "clsloss_f.txt",  "cycloss.txt", "advloss_g.txt",  "clsloss_r.txt",  "recloss.txt"]

loss_files = [log_dir / fname for fname in loss_file_names]

losses = [np.loadtxt(file)[args.begin_iter:] for file in loss_files]
labels = [file.stem for file in loss_files]

print(f"#Iteration: {losses[0].shape[0]}")
for lab, los in zip(labels, losses):
    x = np.arange(los.shape[0])
    plt.plot(x, los, label=lab)
    plt.plot(x[args.dot], los[args.dot], ".", color="red")
plt.legend()
plt.show()


w_adv = 1.0
w_cls = 1.0
w_cyc = 1.0
w_rec = 1.0

gen_loss = w_adv * losses[3] + w_cls * losses[1] + w_cyc * losses[2] + w_rec * losses[5]
cls_loss = losses[4]
advdis_loss = losses[0]

x = np.arange(gen_loss.shape[0])
plt.plot(x, gen_loss, label="generator total loss")
plt.plot(x[args.dot], gen_loss[args.dot], ".", color="red")
plt.plot(x, cls_loss, label="classifier total loss")
plt.plot(x[args.dot], cls_loss[args.dot], ".", color="red")
plt.plot(x, advdis_loss, label="discriminator total loss")
plt.plot(x[args.dot], advdis_loss[args.dot], ".", color="red")
plt.legend()
plt.show()
