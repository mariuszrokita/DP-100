# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

import argparse
import os

from util import print_nicely

print_nicely("In train.py")
print_nicely("As a data scientist, this is where I use my training code.")

parser = argparse.ArgumentParser("train")

parser.add_argument("--input_data", type=str, help="input data")
parser.add_argument("--output_train", type=str, help="output_train directory")

args = parser.parse_args()

print_nicely("Argument 1: %s" % args.input_data)
print_nicely("Argument 2: %s" % args.output_train)

if not (args.output_train is None):
    os.makedirs(args.output_train, exist_ok=True)
    print_nicely("%s created" % args.output_train)
