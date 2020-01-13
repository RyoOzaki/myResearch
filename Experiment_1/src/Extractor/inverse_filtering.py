import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

def coefficients(length, width, padding, weight):
    left_over = width // 2
    right_over = width - left_over - 1
    coeff_wide = np.zeros((length, length + width - 1))
    ident = np.identity(length)
    for i in range(width):
        coeff_wide[:, i:i+length] += ident * weight[i]
    if padding == "same":
        for i in range(left_over):
            coeff_wide[:, left_over] += coeff_wide[:, i]
        for i in range(right_over):
            coeff_wide[:, -right_over-1] += coeff_wide[:, -i-1]
    coeff = coeff_wide[:, left_over:-right_over]
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


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)

parser.add_argument("--mode", default="none", choices=["none", "triangle", "gaussian"])

parser.add_argument("--reduce", default="mean", choices=["mean", "sum"])

parser.add_argument("--triangle_grad", type=float, default=1.0)
parser.add_argument("--triangle_bias", type=float, default=1.0)

parser.add_argument("--gaussian_sigma", type=float, default=1.0)

parser.add_argument("--padding", default="zero", choices=["zero", "same"])
parser.add_argument("--width", type=int, default=5)

args = parser.parse_args()

source_npz = np.load(args.source)
keys = sorted(list(source_npz.keys()))

feature = {}
for key in keys:
    data = source_npz[key]
    length = data.shape[0]
    if args.mode == "none":
        weight = np.ones(args.width) / args.width
    elif args.mode == "triangle":
        weight = triangle_weight(args.width, args.triangle_grad, args.triangle_bias)
    elif args.mode == "gaussian":
        weight = gaussian_weight(args.width, args.gaussian_sigma)

    if args.reduce == "sum":
        weight = np.ones(args.width) * args.width

    coeff = coefficients(length, args.width, args.padding, weight)
    inv_coeff = np.linalg.pinv(coeff)
    feat = np.dot(inv_coeff, data)

    feature[key] = feat

np.savez(args.output, **feature)
