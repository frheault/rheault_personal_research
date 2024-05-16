#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
This script is used to alter a bundle to reach specific minimum dice
coefficient from the original bundle. The script will subsample, trim, cut,
upsample or transform the streamlines until the minimum dice is reached.
(cannot be combined in one run, use the script multiple times if needed)
TODO: DOCSTRING
"""

import argparse
import logging
import os

from dipy.io.streamline import load_tractogram, save_tractogram
import numpy as np

from my_research.utils.tractograms import (transform_streamlines_alter,
                                           trim_streamlines_alter,
                                           cut_streamlines_alter,
                                           subsample_streamlines_alter,
                                           upsample_streamlines_alter)
from my_research.utils.util import _clean_sft


def buildArgsParser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_bundle',
                   help='Input bundle filename. Format must be readable'
                   ' by the Nibabel.')
    p.add_argument('out_bundle',
                   help='Output bundle filename. Format must be readable'
                   ' by the Nibabel.')

    p.add_argument('--min_dice', type=float, default=0.90,
                   help='Minimum dice to reach.')
    p.add_argument('--epsilon', type=float, default=0.01,
                   help='Epsilon for the convergence.')

    p.add_argument('--force_overwrite', '-f', action='store_true',
                   help='Force the overwriting of the output file.')
    p.add_argument('--seed', '-s', type=int, default=None,
                   help='Seed for reproducibility. '
                        'Default based on --min_dice.')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Produce verbose output.')

    g = p.add_argument_group(title='Alteration options')
    g1 = g.add_mutually_exclusive_group(required=True,)
    g1.add_argument('--subsample', action='store_true',
                    help='Subsample the streamlines.')
    g1.add_argument('--trim', action='store_true',
                    help='Trim the streamlines.')
    g1.add_argument('--cut', action='store_true',
                    help='Cut the streamlines.')
    g1.add_argument('--upsample', action='store_true',
                    help='Upsample the streamlines.')
    g1.add_argument('--transform', action='store_true',
                    help='Transform the streamlines.')
    g.add_argument('--from_end', action='store_true',
                   help='Cut from the other end of the streamlines.')

    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    if not os.path.isfile(args.in_bundle):
        parser.error(f'{args.in_bundle} must be a file!')

    if os.path.exists(args.out_bundle) and not args.force_overwrite:
        parser.error(f'{args.out_bundle} is already a file, use -f to enable '
                     'the overwrite.')
    if args.from_end and args.cut is None:
        parser.error('The --from_end option is only available with --cut.')

    if args.seed is None:
        np.random.seed(int(args.min_dice * 1000))
    else:
        np.random.seed(args.seed)

    if args.verbose:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    sft = load_tractogram(args.in_bundle, 'same')

    if args.subsample:
        altered_sft = subsample_streamlines_alter(sft, args.min_dice,
                                                  epsilon=args.epsilon)
    elif args.trim:
        altered_sft = trim_streamlines_alter(sft, args.min_dice,
                                             epsilon=args.epsilon)
    elif args.cut:
        altered_sft = cut_streamlines_alter(sft, args.min_dice,
                                            epsilon=args.epsilon,
                                            from_start=not args.from_end)
    elif args.upsample:
        altered_sft = upsample_streamlines_alter(sft, args.min_dice,
                                                 epsilon=args.epsilon)
    elif args.transform:
        altered_sft = transform_streamlines_alter(sft, args.min_dice,
                                                  epsilon=args.epsilon)
    altered_sft = _clean_sft(altered_sft)

    save_tractogram(altered_sft, args.out_bundle)


if __name__ == "__main__":
    main()
