#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TODO
"""
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

from my_research.utils.coloring import (select_dissimilar_colors,
                                        compute_cielab_distances,
                                        compute_bundle_distance_matrix,
                                        greedy_coloring,
                                        plot_scatter_with_colors)
from dipy.io.streamline import save_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_processes_arg,
                             add_verbose_arg,
                             add_reference_arg,
                             assert_headers_compatible,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             assert_output_dirs_exist_and_empty,
                             validate_nbr_processes)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_tractograms', nargs='+',
                   help='Input tractograms (.trk or .tck).')
    p.add_argument('out_dir',
                   help='Output directory.')
    p.add_argument('--out_LUT',
                   help='Save a Look-Up Table file (.json).')
    p.add_argument('--out_palette',
                   help='Save the color palette image (.png).')
    p.add_argument('--coloring_method', default='most_different_sum',
                   choices=['first_available', 'most_different_sum',
                            'most_different_mean', 'most_different_min',
                            'most_optimal'],
                   help="Method to use for coloring [%(default)s].")
    p.add_argument('--nb_color', type=int, default=20,
                   help="Number of dissimilar colors (palette) to generate "
                        "[%(default)s].")
    
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_json_args(p)

    return p


def plot_colors(rgb_colors):
    """Plot RGB colors in a grid layout."""
    num_colors = len(rgb_colors)
    num_cols = int(np.ceil(np.sqrt(num_colors)))
    num_rows = int(np.ceil(num_colors / num_cols))

    plt.figure(figsize=(num_cols, num_rows))
    for i, color in enumerate(rgb_colors):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow([[color]])
        plt.axis('off')
    plt.show()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractograms)
    assert_output_dirs_exist_and_empty(parser, args, args.out_dir,
                                       create_dir=True)
    assert_outputs_exist(parser, args, [], [args.out_LUT, args.out_palette])
    assert_headers_compatible(parser, args.in_tractograms)

    np.random.seed(0)

    rgb_colors = select_dissimilar_colors(
        args.nb_color, h_range=(0, 1), s_range=(0.5, 1), v_range=(0.7, 1))
    # plot_colors(rgb_colors)

    sft_lists = [load_tractogram_with_reference(parser, args, filename)
                 for filename in args.in_tractograms]
    NB_SAMPLE = len(sft_lists)
    colors_repeat = np.repeat(rgb_colors, np.ceil(
        NB_SAMPLE / args.nb_color), axis=0)[:NB_SAMPLE]

    indices = np.arange(NB_SAMPLE)
    np.random.shuffle(indices)
    # print(sft_lists)
    sft_lists = [sft_lists[i] for i in indices]
    filenames = [args.in_tractograms[i] for i in indices]

    distance_matrix = compute_bundle_distance_matrix(sft_lists,
                                                     distance='bundle_adjacency')
    print(distance_matrix)
    # MAX_DIST = np.std(distance_matrix)
    MAX_DIST = np.std(distance_matrix)
    print(MAX_DIST, '0')
    # Plot histogram of distance
    # plt.hist(distance_matrix.flatten(), bins=20)
    # plt.show()
    ordering = greedy_coloring(distance_matrix, rgb_colors,
                               max_distance=MAX_DIST,
                               coloring_method=args.coloring_method)
    print(ordering)
    rgb_colors = rgb_colors[ordering]
    # Save tractograms
    for sft, filename, rgb_color in zip(sft_lists, filenames, rgb_colors):
        basename = os.path.basename(filename)
        out_filename = f'{args.out_dir}/{basename}'
        red, green, blue = rgb_color * 255

        tmp = [np.tile([red, green, blue], (len(i), 1))
               for i in sft.streamlines]
        sft.data_per_point['color'] = tmp
        save_tractogram(sft, out_filename)

    # Save LUT
    if args.out_LUT:
        lut = {filename: rgb_colors[i].tolist()
               for i, filename in enumerate(filenames)}
        with open(args.out_LUT, 'w') as f:
            json.dump(lut, f, indent=args.indent, sort_keys=args.sort_keys)

    # plot_scatter_with_colors(coordinates, rgb_colors[ordering])


if __name__ == '__main__':
    main()
