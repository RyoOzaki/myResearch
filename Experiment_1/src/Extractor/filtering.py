import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

def coefficients(length, weight):
    width = weight.shape[0]
    left_over = width // 2
    right_over = width - left_over - 1
    coeff_wide = np.zeros((length, length + width - 1))
    ident = np.identity(length)
    for i in range(width):
        coeff_wide[:, i:i+length] += ident * weight[i]
    coeff = coeff_wide[:, left_over:-right_over]
    coeff /= coeff.sum(axis=1, keepdims=True)
    return coeff

def gaussian_weight(width, sigma):
    left_over = width // 2
    x = np.arange(width, dtype=float) - left_over
    x -= x.mean()
    weight = np.exp(-(x**2)/(2 * sigma**2)) / (2 * np.pi * sigma**2)
    weight /= weight.sum()
    return weight

def triangle_weight(width, grad, bias):
    left_over = width // 2
    x = np.arange(width, dtype=float) - left_over
    x -= x.mean()
    weight = np.zeros(width)
    weight[:left_over] = grad * (x[:left_over] - x[0]) + bias
    weight[left_over:] = -grad * (x[left_over:] - x[-1]) + bias
    weight /= weight.sum()
    return weight

def moving_averave_weight(width):
    if width % 2 == 0:
        weight = np.ones(width+1)
        weight[0] = 0.5
        weight[-1] = 0.5
    else:
        weight = np.ones(width)
    weight /= weight.sum()
    return weight


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path)

parser.add_argument("--mode", default="none", choices=["none", "triangle", "gaussian"])

parser.add_argument("--triangle_grad", type=float, default=1.0)
parser.add_argument("--triangle_bias", type=float, default=1.0)

parser.add_argument("--gaussian_sigma", type=float, default=1.0)

parser.add_argument("--width", type=int, default=5)

parser.add_argument("--scale", type=int, default=1.0)

args = parser.parse_args()

source_npz = np.load(args.source)
keys = sorted(list(source_npz.keys()))

if args.output is None:
    out_file = args.source.with_name(f"filt_{args.source.stem}.npz")
else:
    out_file = args.output
out_file.parent.mkdir(exist_ok=True, parents=True)

feature = {}
for key in keys:
    data = source_npz[key]
    length = data.shape[0]
    if args.mode == "none":
        weight = moving_averave_weight(args.width)
    elif args.mode == "triangle":
        weight = triangle_weight(args.width, args.triangle_grad, args.triangle_bias)
    elif args.mode == "gaussian":
        weight = gaussian_weight(args.width, args.gaussian_sigma)

    coeff = coefficients(length, weight)
    feat = np.dot(coeff, data) * args.scale

    feature[key] = feat

np.savez(out_file, **feature)
