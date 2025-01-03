#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import argparse
from itertools import product
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree


def compute_bundle_adjacency_voxel(binary_1, binary_2, non_overlap=False):
    """
    Compute the distance in millimeters between two bundles in the voxel
    representation. Convert the bundles to binary masks. Each voxel of the
    first bundle is matched to the the nearest voxel of the second bundle and
    vice-versa.
    Distance between matched paired is averaged for the final results.
    Parameters
    ----------
    binary_1: ndarray
        Binary mask computed from the first bundle
    binary_2: ndarray
        Binary mask computed from the second bundle
    non_overlap: bool
        Exclude overlapping voxels from the computation.
    Returns
    -------
    float: Distance in millimeters between both bundles.
    """
    b1_ind = np.argwhere(binary_1 > 0)
    b2_ind = np.argwhere(binary_2 > 0)
    b1_tree = cKDTree(b1_ind)

    distance_1, _ = b1_tree.query(b2_ind)

    if non_overlap:
        non_zeros_1 = np.nonzero(distance_1)
        if not non_zeros_1[0].size == 0:
            distance_b1 = np.mean(distance_1[non_zeros_1])
        else:
            distance_b1 = 0
    else:
        distance_b1 = np.mean(distance_1)

    return distance_b1


def compute_dice_voxel(density_1, density_2):
    """
    Compute the overlap (dice coefficient) between two
    density maps (or binary).

    Parameters
    ----------
    density_1: ndarray
        Density (or binary) map computed from the first bundle
    density_2: ndarray
        Density (or binary) map computed from the second bundle

    Returns
    -------
    A tuple containing:

    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles.
    - float: Value between 0 and 1 that represent the spatial aggrement
        between both bundles, weighted by streamlines density.
    """
    overlap_idx = np.nonzero(density_1 * density_2)
    numerator = 2 * len(overlap_idx[0])
    denominator = np.count_nonzero(density_1) + np.count_nonzero(density_2)

    if denominator > 0:
        dice = numerator / float(denominator)
    else:
        dice = np.nan

    overlap_1 = density_1[overlap_idx]
    overlap_2 = density_2[overlap_idx]
    w_dice = np.sum(overlap_1) + np.sum(overlap_2)
    denominator = np.sum(density_1) + np.sum(density_2)
    if denominator > 0:
        w_dice /= denominator
    else:
        w_dice = np.nan

    return dice, w_dice



def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', metavar='IN_FILE',
                   help='Input file name, in nifti format.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    img = nib.load(args.in_image)
    data = img.get_fdata()


    labels = np.unique(data).astype(np.uint16)
    labels = labels[labels > 0]
    def label_to_binary_mask(array, label):
        arr = np.zeros(array.shape, dtype=np.uint8)
        arr[array == label] = 1
        return arr
    
    masks = {label: label_to_binary_mask(data, label) for label in labels}
    flipped_masks = {label: np.flip(mask, axis=0) for label, mask in masks.items()}
    
    # Compute overlaps and store results
    results = {}
    for label1, label2 in product(labels, labels):

        if label1 not in results:
            results[label1] = {}
        print(label1, label2)
        overlap_score = compute_bundle_adjacency_voxel(flipped_masks[label1], masks[label2])
        results[label1][label2] = overlap_score
    
    # Identify symmetrical pairs
    symmetrical_pairs = []
    used_labels = set()
    for label1 in labels:
        min_score = 999
        last_win = 0, 0
        for label2 in labels:
            print(label1, label2)
            if label1 not in used_labels and label2 not in used_labels:
                score1 = results[label1].get(label2, 0)
                score2 = results[label2].get(label1, 0)
                # Define symmetry criterion here (e.g., score difference within a certain range)
                if (score1 + score2) / 2.0 < 10 \
                    and (score1 + score2) / 2.0 < min_score \
                    and abs(score1 - score2) < 5:  # Example symmetry criterion
                    last_win = (label1, label2)
                    min_score = (score1 + score2) / 2.0
        symmetrical_pairs.append(last_win + (min_score,))
        used_labels.update(last_win)

    # Sort and select top 3 matches
    for label in labels:
        results[label] = sorted(results[label].items(), key=lambda x: x[1], reverse=False)[:3]
    
    # Pretty print results
    for label, matches in results.items():
        print(f"Label {label} top matches: {matches}")


    # Display symmetrical pairs
    print("Symmetrical pairs:")
    for pair in symmetrical_pairs:
        print(f"Pair: {pair[0]} - {pair[1]}, Score: {pair[2]}")

# Example array initialization and function call, if needed for demonstration
# array = np.array([...])  # Example array initialization
# find_top_matches(array)



if __name__ == "__main__":
    main()
