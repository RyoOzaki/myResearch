import numpy as np
import matplotlib.pyplot as plt
import readline
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--target", type=Path, required=True)

args = parser.parse_args()

target_npz = np.load(args.target)
candidates = list(target_npz.keys())

def completer(text, state):
    options = [i for i in candidates if i.startswith(text)]
    if state < len(options):
        return options[state]
    else:
        return None

readline.parse_and_bind("tab: complete")
readline.set_completer(completer)

before_key = None
while True:
    key = input(">> ")
    if key == "exit":
        break
    elif before_key and key == "save":
        plt.plot(target_npz[before_key])
        plt.title(before_key)
        plt.savefig(f"{before_key.replace('/', '_')}.png")
        plt.clf()
    elif key in target_npz:
        print(f"shape : {target_npz[key].shape}")
        plt.plot(target_npz[key])
        plt.title(key)
        plt.show()
        plt.clf()
        before_key = key
