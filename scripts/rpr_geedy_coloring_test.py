#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script generates a grid or random set of samples, assigns colors using
various coloring strategies, and evaluates the color distances between 
neighboring points. The script provides a visual representation of the 
colored points.

Dependencies:
- numpy
- matplotlib
- scikit-learn
- scikit-image

Usage:
    python rpr_geedy_coloring_test.py --sample_gen <grid|random|cluster> \
                                      --coloring_method <first_available|most_different_sum|most_different_mean|most_different_min|most_optimal> \
                                      --nb_sample <int> --nb_color <int> --nb_cluster <int>
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from sklearn.cluster import KMeans

from my_research.utils.coloring import (
    select_dissimilar_colors,
    compute_cielab_distances,
    compute_euclidean_distance_matrix,
    greedy_coloring,
    plot_scatter_with_colors,
    generate_2d_grid_coordinates,
    generate_clusters
)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--sample_gen', required=True,
                   choices=['grid', 'random', 'cluster'],
                   help="Method to generate sample points [%(default)s].")
    p.add_argument('--coloring_method', default='most_different_sum',
                   choices=['first_available', 'most_different_sum',
                            'most_different_mean', 'most_different_min',
                            'most_optimal'],
                   help="Method to use for coloring [%(default)s].")
    p.add_argument('--nb_sample', type=int, default=100,
                   help="Number of samples to generate [%(default)s].")
    p.add_argument('--nb_color', type=int, default=20,
                   help="Number of dissimilar colors to generate "
                        "[%(default)s].")
    p.add_argument('--nb_cluster', default=None,
                   help="Number of clusters to generate (only used when "
                   "--sample_gen cluster) [%(default)s].")
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


def compute_max_distance(coordinates, nb_color, sample_gen, nb_cluster):
    """Compute maximum distance based on the sample generation method."""
    if sample_gen == 'grid':
        distance_matrix = compute_euclidean_distance_matrix(coordinates)
        max_dist = 1 * int(np.sqrt(nb_color))
    elif sample_gen == 'random':
        tree = KDTree(coordinates)
        closest_dist, _ = tree.query(coordinates, k=nb_color)
        max_dist = np.mean(closest_dist[closest_dist > 0])
    elif sample_gen == 'cluster':
        max_dist = 0
        kmeans = KMeans(n_clusters=nb_cluster, n_init=10)
        kmeans.fit(coordinates)
        centers = kmeans.cluster_centers_
        for i, cluster_center in enumerate(centers):
            curr_coord = coordinates[kmeans.labels_ == i]
            dist_to_all = compute_euclidean_distance_matrix(curr_coord)
            dist_to_all = dist_to_all[dist_to_all > 0]
            max_dist = max(max_dist, np.max(dist_to_all))
        max_dist *= 2.0
    else:
        raise ValueError(f"Unknown sample generation method: {sample_gen}")

    return max_dist


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.nb_cluster is not None and args.nb_cluster > args.nb_sample:
        raise ValueError("Number of clusters cannot exceed number of samples.")
    if args.nb_cluster and not args.sample_gen == 'cluster':
        raise ValueError("Number of clusters can only be used with "
                         "--sample_gen cluster.")

    np.random.seed(0)

    rgb_colors = select_dissimilar_colors(
        args.nb_color, h_range=(0, 1), s_range=(0.5, 1), v_range=(0.7, 1))
    plot_colors(rgb_colors)

    colors_repeat = np.repeat(rgb_colors, np.ceil(
        args.nb_sample / args.nb_color), axis=0)[:args.nb_sample]

    if args.sample_gen == 'grid':
        coordinates = generate_2d_grid_coordinates(args.nb_sample)
    elif args.sample_gen == 'random':
        coordinates = np.random.rand(
            args.nb_sample * 2).reshape((args.nb_sample, 2))
    elif args.sample_gen == 'cluster':
        coordinates, _ = generate_clusters(
            args.nb_cluster, args.nb_sample // args.nb_cluster, 0.01, 0)
    else:
        raise ValueError(
            f"Unknown sample generation method: {args.sample_gen}")

    max_dist = compute_max_distance(
        coordinates, args.nb_color, args.sample_gen, args.nb_cluster)

    np.random.shuffle(coordinates)
    distance_matrix = compute_euclidean_distance_matrix(coordinates)
    print(distance_matrix)
    plot_scatter_with_colors(coordinates, colors_repeat)

    ordering = greedy_coloring(distance_matrix, rgb_colors,
                               max_distance=max_dist, coloring_method=args.coloring_method)

    focus_idx = 0
    focus_color = rgb_colors[ordering[focus_idx]]
    neighbors = distance_matrix[focus_idx] < max_dist
    neighbors_idx = np.where(neighbors)[0]
    neighbor_colors = rgb_colors[ordering[neighbors]]

    color_distances = compute_cielab_distances(neighbor_colors, [focus_color])
    total_color_distance = np.sum(color_distances)
    for i, d in enumerate(color_distances):
        print(f'Color distance {neighbors_idx[i]}: {d}')
    print(f'Sum of color distances: {total_color_distance}')

    print(f'Number of unique colors: {len(set(ordering))}')
    plot_scatter_with_colors(coordinates, rgb_colors[ordering])


if __name__ == '__main__':
    main()
