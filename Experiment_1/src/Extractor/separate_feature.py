import numpy as np
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--source", type=Path, required=True, help="feature file to be separated (npz)")
parser.add_argument("--show_dimension", action="store_true", help="show the dimension of feature and exit")
parser.add_argument("--prefix", default="", help="prefix of name, you can use specific option #b and #e")
parser.add_argument("--single_prefix", default="", help="prefix of name when #b==#e, you can use specific option #b")
parser.add_argument("--suffix", default="(#b-#e)", help="prefix of name, you can use specific option #b and #e")
parser.add_argument("--single_suffix", default="(#b)", help="suffix of name when #b==#e, you can use specific option #b")

parser.add_argument("--recipe", required=True, nargs="+", type=int, help="recipe to separate dimension how to, you can use 0 to the remaining number (e.g., input [5] -> specify [1 2 0] -> parse as [1 2 2])")

args = parser.parse_args()

source = args.source
npz_obj = np.load(source)
keys = list(npz_obj.keys())
source_dim = npz_obj[keys[0]].shape[1]

if args.show_dimension:
    print(f"Dimension of {source} is {source_dim}.")
    exit(0)

if args.recipe is None:
    raise ValueError("--recipe option must be specified")
    exit(1)

recipe = np.array(args.recipe, dtype=int)
if (recipe == 0).sum() > 1:
    raise ValueError("option value 0 in --recipe can be specified only once or not specified.")
    exit(2)
recipe[recipe == 0] = source_dim - recipe.sum()

if not np.all(recipe >= 0):
    raise ValueError(f"you can not use 0 or minus values in --recipe option, recipe={recipe}")
    exit(3)

if recipe.sum() != source_dim:
    raise ValueError(f"sum of recipe must be same as features dimension of source file, source dim={source_dim}, sum of recipe={recipe.sum()}")
    exit(4)

cumsum_recipe = np.concatenate(([0], np.cumsum(recipe)))

for b, e in zip(cumsum_recipe[:-1], cumsum_recipe[1:]):
    if b == e-1:
        prefix = args.single_prefix.replace("#b", str(b))
        suffix = args.single_suffix.replace("#b", str(b))
    else:
        prefix = args.prefix.replace("#b", str(b)).replace("#e", str(e-1))
        suffix = args.suffix.replace("#b", str(b)).replace("#e", str(e-1))
    output_file = source.with_name(f"{prefix}{source.stem}{suffix}.npz")
    output_dict = {}
    for key in keys:
        output_dict[key] = npz_obj[key][:, b:e]
    np.savez(output_file, **output_dict)
