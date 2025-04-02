import itertools
from time import time
import tqdm
import warnings
import nibabel as nib
from scipy.spatial.distance import cdist
import glob
import os
import sys
import scipy.io
import numpy as np
import multiprocessing
import argparse
import logging
from scipy.spatial import KDTree

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


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_assignement_files', nargs=2,
                   metavar='ASSIGN_FILES',
                   help='Both assignment files:\n'
                        '\t1. MAT file with all possible connectivity signatures\n'
                        '\t2. MAT file with mapping of original to new labels.')
    p.add_argument('in_dir',
                   help='Input directory containing subject decomposed '
                        'TDI files.')
    p.add_argument('in_wm_mask',
                   help='Input WM mask file.')
    p.add_argument('in_nufo',
                   help='Input NUFO file.')
    p.add_argument('out_labels',
                   help='Output directory to save the results.')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # all_labels = scipy.io.loadmat(ALL_LABELS_FILENAME)
    mapping_labels = scipy.io.loadmat(args.in_assignement_files[1])
    orig_labels = np.squeeze(mapping_labels['orig_label'])
    new_labels = np.squeeze(mapping_labels['new_label'])
    mapping_labels = dict(zip(orig_labels, new_labels))

    # Define LABEL matrix (ensure dtype is appropriate, e.g., int)
    # Using the second LABEL matrix provided in the MATLAB code
    labels = np.ones((15, 15), dtype=int) * -1
    comb_list = np.triu_indices(15, k=0)
    for i, coord in enumerate(zip(*comb_list)):
        labels[coord] = i + 1

    # --- Main Loop ---

    thresh = 10
    print(f"--- Processing threshold: {thresh} ---")

    # Check if the file exists
    mat_data = scipy.io.loadmat(args.in_assignement_files[0])
    B = mat_data['B']

    # Prepare TOMATCH (Python uses 0-based indexing, so start from column 1)
    # Ensure B has at least 2 columns
    if B.shape[1] < 2:
        raise ValueError(f"Error: Matrix 'B' loaded from {args.in_assignement_files[0]} "
                         "has fewer than 2 columns.")
    global all_signatures, all_signatures_dict
    all_signatures = B[:, 1:].astype(np.uint8)
    all_signatures_dict = {hash(tuple(row)): i for i, row in enumerate(all_signatures)}

    # Get the first column of B for finding indices
    tmp = B[:, 0]

    # Find the first and last indices for NUFO groups (AA, BB, CC)
    indices_1 = np.where(tmp == 1)[0]
    first_index_1, last_index_1 = indices_1[0], indices_1[-1]

    indices_2 = np.where(tmp == 2)[0]
    first_index_2, last_index_2 = indices_2[0], indices_2[-1]

    indices_3 = np.where(tmp == 3)[0]
    first_index_3, last_index_3 = indices_3[0], indices_3[-1]

    # Define index ranges (inclusive for slicing) # TODO Overlap?
    # AA_slice = slice(first_index_1, last_index_1 + 1)
    # BB_slice = slice(first_index_2, last_index_2 + 1)
    # CC_slice = slice(first_index_3, last_index_3 + 1)
    # print(f"Index ranges: AA={AA_slice}, BB={BB_slice}, CC={CC_slice}")

    # Check if output file already exists
    if os.path.isfile(args.out_labels) and not args.overwrite:
        raise ValueError(f"Output file exists: {args.out_labels}. SKIPPING...")

    if not os.path.isfile(args.in_nufo):
        raise ValueError(
            f"NUFO file not found: {args.in_nufo}. Skipping subject.")

    if not os.path.isfile(args.in_wm_mask):
        raise ValueError(f"WM file not found: {args.in_wm_mask}. Skipping subject.")

    nufo_img = nib.load(args.in_nufo)
    nufo_data = nufo_img.get_fdata().astype(np.int32)
    nufo_data = np.clip(nufo_data, 0, 3)

    wm_img = nib.load(args.in_wm_mask)
    wm_data = wm_img.get_fdata().astype(np.float32)
    wm_mask = wm_data > 0.0

    max_label = np.max(labels[labels > 0])
    tdi_data = np.zeros(wm_data.shape + (max_label,), dtype=np.uint8)
    print(f"Initialized labels array with shape: {tdi_data.shape}")

    print("Grabbing TDI files...")
    count = 0
    comb_list = np.triu_indices(15, k=0)
    for id_1, id_2 in tqdm.tqdm(zip(*comb_list), total=len(comb_list[0])):
        tdi_path = os.path.join(args.in_dir, f'{id_1+1}_{id_2+1}.nii.gz')

        if not os.path.isfile(tdi_path):
            print(f"Warning: TDI file not found: {tdi_path}")
            continue

        img = nib.load(tdi_path)
        data = img.get_fdata().astype(np.float32)

        # Binarize it based on the threshold
        data[data < thresh] = 0.0
        data[data >= thresh] = 1.0

        label_index = labels[id_1, id_2] - 1
        tdi_data[..., label_index] = data
        count += 1

    if count != max_label:
        print(
            f"Warning: Expected {max_label - 1} TDI files, but found {count}.")

    def _process_voxel(signature):
        """
        Process a single voxel's signature against the NUFO signatures.
        """
        global all_signatures, all_signatures_dict
        if np.sum(signature) == 0:
            print("Skipping empty signature")
            return 0

        curr_hash = hash(tuple(signature))
        # Check if the current signature is in the dictionary
        if curr_hash in all_signatures_dict:
            best_match = all_signatures_dict[curr_hash]
            return best_match + 1
        return 0
        # else:
        #     # Calculate distances (Cityblock = Manhattan)
        #     # signature not in the dictionary, so compute distances
        #     D = cdist(signature.reshape(1, -1), all_signatures,
        #             metric='cityblock')[0]

        #     best_match = np.argmin(D)
        #     min_val = D[best_match]

        # # Check if the minimum value is too high
        # if min_val >= 10:
        #     return -1

        # return best_match + 1 if best_match else 0

    # Flatten except the last dimension
    intersection_mask = np.sum(tdi_data, axis=-1) > 0 & wm_mask
    tdi_data = tdi_data[intersection_mask]
    nufo_data = nufo_data[intersection_mask]
    num_voxels = np.count_nonzero(intersection_mask)
    labels_ravel = np.zeros_like(nufo_data, dtype=np.int32)

    # Voxel-wise processing of signatures
    for pos in tqdm.tqdm(range(num_voxels), total=num_voxels):
        curr_signature = tdi_data[pos]
        best_match = _process_voxel(curr_signature)

        # Check if the result is valid
        if best_match == -1:
            continue
        
        labels_ravel[pos] = mapping_labels.get(best_match, -2)
        # labels_ravel[pos] = best_match

    labels = np.zeros_like(wm_data, dtype=np.int32)
    labels[intersection_mask] = labels_ravel
    coord_unfound = np.argwhere(labels == -2)
    coord_found = np.argwhere(labels > 0)
    tree = KDTree(coord_found)
    _, idx = tree.query(coord_unfound, k=1, distance_upper_bound=5)

    # Filter out invalid indices (e.g., those that exceed the length of coord_found)
    valid_idx_mask = idx < len(coord_found)
    # valid_idx = idx[valid_idx_mask]
    invalid_idx = ~valid_idx_mask

    # Extract the labels at the neighbor coordinates
    labels[coord_unfound[valid_idx_mask, 0],
           coord_unfound[valid_idx_mask, 1],
           coord_unfound[valid_idx_mask, 2]] = new_labels

    #deal with the invalid indexes.
    labels[coord_unfound[invalid_idx, 0],
           coord_unfound[invalid_idx, 1],
           coord_unfound[invalid_idx, 2]] = -1

    print(f"Saving labels to: {args.out_labels}")
    nib.save(nib.Nifti1Image(labels, wm_img.affine), args.out_labels)


if __name__ == '__main__':
    main()
