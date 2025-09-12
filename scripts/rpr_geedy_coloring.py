#!/usr/bin/env python
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
import logging

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
                             load_matrix_in_any_format)

from scilpy.image.volume_operations import register_image
import nibabel as nib
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import transform_streamlines
from scilpy.tractograms.tractogram_operations import flip_sft


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
                            'most_optimal', 'most_similar'],
                   help="Method to use for coloring [%(default)s].")
    p.add_argument('--nb_color', type=int, default=20,
                   help="Number of dissimilar colors (palette) to generate "
                        "[%(default)s].")
    p.add_argument('--enable_symmetry', metavar='TRANSFORMATION',
                   help='Enable L/R symmetry in the coloring using a '
                        'transformation matrix to a MNI152 standard space.')
    p.add_argument('--source_anat',
                   help='Path of the reference file (.nii or nii.gz).')
    p.add_argument('--target_anat',
                   help='Path to the target anat for registration. '
                        'must be close to ACPC space.')
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

    if args.enable_symmetry and not (args.target_anat and args.source_anat):
        parser.error('You must provide a target template with --target_anat.')

    args.in_tractograms = sorted(args.in_tractograms)
    np.random.seed(0)

    rgb_colors = select_dissimilar_colors(
        args.nb_color, h_range=(0, 1), s_range=(0.75, 1), v_range=(0.75, 1))

    sft_list = [load_tractogram_with_reference(parser, args, filename)
                for filename in args.in_tractograms]
    NB_SAMPLE = len(sft_list)
    indices = np.arange(NB_SAMPLE)
    # if not args.enable_symmetry:
    np.random.shuffle(indices)

    sft_list = [sft_list[i] for i in indices]
    filenames = [args.in_tractograms[i] for i in indices]

    if args.enable_symmetry:
        matched = {}
        flip_sft_list = []
        transformation = load_matrix_in_any_format(args.enable_symmetry)
        logging.debug('Evaluation symmetry')
        for i, (filename, sft) in enumerate(zip(filenames, sft_list)):
            # if i in matched.values() or i in matched.keys():
            #     continue

            streamlines = sft.streamlines.copy()
            streamlines = transform_streamlines(streamlines,
                                                np.linalg.inv(
                                                    transformation),
                                                in_place=False)

            new_sft = StatefulTractogram(streamlines,
                                        args.target_anat,
                                        Space.RASMM)
            new_sft = flip_sft(new_sft, ['x'])
            streamlines = new_sft.streamlines.copy()

            streamlines = transform_streamlines(streamlines,
                                                transformation,
                                                in_place=False)
            new_sft = StatefulTractogram(streamlines,
                                        args.source_anat,
                                        Space.RASMM)
            flip_sft_list.append(new_sft)
        distance_matrix = compute_bundle_distance_matrix(sft_list,
                                                         symmetry=flip_sft_list,
                                                            distance='dice',
                                                            use_mean=True,
                                                            disable_tqdm=False)
        print(distance_matrix)
        for index, value in enumerate(distance_matrix):
            sorted_index = np.argsort(value)
            best_match = int(sorted_index[0])

            if best_match not in matched.values() and best_match not in matched.keys():
                # Verify left vs right
                avg_coords_1 = np.mean(sft_list[index].streamlines._data, axis=0)
                avg_coords_2 = np.mean(sft_list[best_match].streamlines._data, axis=0)
                print(f'{index} -> {best_match}')
                print(f'{filenames[index]} -> {filenames[best_match]}')
                if avg_coords_1[0] > avg_coords_2[0]:
                    best_match, index = index, best_match
                matched[index] = best_match
        symmetric_sft_list = [sft_list[i] for i in matched.keys()]
        logging.getLogger().setLevel(args.verbose)

        distance_matrix = compute_bundle_distance_matrix(symmetric_sft_list,
                                                         distance='bundle_adjacency')
    else:
        logging.getLogger().setLevel(args.verbose)

        distance_matrix = compute_bundle_distance_matrix(sft_list,
                                                         distance='bundle_adjacency')
    MAX_DIST = np.std(distance_matrix)

    ordering = greedy_coloring(distance_matrix, rgb_colors,
                               max_distance=MAX_DIST,
                               coloring_method=args.coloring_method)

    if args.enable_symmetry:
        new_ordering = np.zeros(len(sft_list), dtype=int)
        for i, (key, value) in enumerate(matched.items()):
            new_ordering[key] = ordering[i]
            new_ordering[value] = ordering[i]
        ordering = new_ordering
    rgb_colors = rgb_colors[ordering]


    # Save tractograms
    for sft, filename, rgb_color in zip(sft_list, filenames, rgb_colors):
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


if __name__ == '__main__':
    main()
