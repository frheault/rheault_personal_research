#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Display an histogram of intensities.
"""

import argparse
import os

import nibabel as nib

from my_research.utils.image import display_histogram


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', metavar='IN_FILE',
                   help='Input file name, in nifti format.')
    p.add_argument('--log', action='store_true',
                   help='Display the log of probabilities.')
    p.add_argument('--nb_bins', type=int,
                   help='How many bins to use.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not os.path.isfile(args.in_image):
        raise IOError('{} does not exist!'.format(args.in_image))

    data = nib.load(args.in_image).get_fdata()
    display_histogram(data, log=args.log, nb_bins=args.nb_bins)


if __name__ == "__main__":
    main()
