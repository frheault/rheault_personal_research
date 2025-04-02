#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, label, maximum_filter, binary_dilation, binary_erosion, binary_fill_holes, generate_binary_structure, gaussian_filter
from skimage.segmentation import watershed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import KDTree
from skimage.morphology import skeletonize_3d, thin
import cv2


def gamma_correction(image, gamma=1.0):
    """
    Apply Gamma Correction to the input image.
    
    Parameters
    ----------
    image : numpy.ndarray
        The input image data. Shape (height, width, 3).
    gamma : float, optional
        The gamma value to adjust the image brightness.
        > 1: darken, < 1: brighten, = 1: no change.

    Returns
    -------
    numpy.ndarray
        The gamma-corrected image.
        
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    corrected = cv2.LUT(image, table)
    return corrected

def segment_lesions(mask_data, img_data, smoothness_factor=0.0, compactness_factor=0.0,
                    affine=np.eye(4)):
    """
    Segments potential lesions from a series of masks.

    Args:
        mask_files (list of str): List of paths to the mask NIfTI files.
        smoothness_factor (float): Standard deviation for Gaussian smoothing of the distance map.
        compactness_factor (float): Compactness parameter for the watershed algorithm.

    Returns:
        nib.Nifti1Image: The labeled image of the segmented lesions.
    """

    # Remove all connected components that are less than 6 voxels
    labeled_mask, num_features = label(mask_data)
    sizes = np.bincount(labeled_mask.ravel())
    for i in range(1, num_features + 1):
        if sizes[i] < 6:
            mask_data[labeled_mask == i] = 0

    struct = generate_binary_structure(3, 1)
    mask_data = binary_fill_holes(mask_data)
    mask_data = binary_dilation(mask_data, structure=struct, iterations=1)
    mask_orig = mask_data.copy()
    mask_data = binary_dilation(mask_data, structure=struct, iterations=1)

    distance_map = distance_transform_edt(mask_data, sampling=1)
    # gradient_x, gradient_y, gradient_z = np.gradient(img_data)

    # # You can also calculate the magnitude of the gradient
    # gradient_map = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
    # gradient_map *= mask_data
    # gradient_map -= gradient_map.min()
    # gradient_map /= gradient_map.max()

    distance_map = mask_data * img_data
    # distance_map -= distance_map.min()
    # distance_map /= distance_map.max()
    distance_map -= distance_map.min()
    distance_map /= np.percentile(distance_map[distance_map > 0], 90)
    distance_map[distance_map < np.percentile(distance_map[distance_map > 0], 99)] **= 2
    if smoothness_factor > 0:
        distance_map = gaussian_filter(distance_map, sigma=smoothness_factor)

    nib.save(nib.Nifti1Image(distance_map, affine), 'distance_map.nii.gz')
    # nib.save(nib.Nifti1Image(gradient_map, affine), 'gradient_map.nii.gz')

    # Define a local neighborhood for finding maxima
    neighborhood = np.ones((7, 7, 7))
    local_max = maximum_filter(distance_map, footprint=neighborhood)
    peaks = (distance_map == local_max)
    peaks &= mask_data
    print(f"Number of initial peaks (potential lesions): {np.count_nonzero(peaks)}")

    markers, num_peaks = label(peaks)
    print(f"Number of identified peaks (potential lesions) before clustering: {num_peaks}")
    cluster = False
    if cluster:
        # Get the 3D coordinates of the peaks
        peak_coords = np.argwhere(peaks)

        peak_cluster_map = np.zeros_like(markers)
        if peak_coords.size > 0:
            # Try different numbers of clusters for k-means
            range_n_clusters = range(2, num_peaks, num_peaks // 100)  # Adjust the upper limit as needed
            silhouette_scores = []
            kmeans_models = {}

            previous_high_score = -1e3
            nb_iterations_without_improvement = 0
            for n_clusters in range_n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(peak_coords)
                silhouette_avg = silhouette_score(peak_coords, cluster_labels)
                if silhouette_avg < previous_high_score:
                    nb_iterations_without_improvement += 1
                else:
                    previous_high_score = silhouette_avg
                    nb_iterations_without_improvement = 0
                if nb_iterations_without_improvement > 20:
                    break
                silhouette_scores.append(silhouette_avg)
                kmeans_models[n_clusters] = kmeans
                print(f"\tFor n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}")

            # Suggest the optimal number of clusters based on the highest silhouette score
            if silhouette_scores:
                optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
                print(f"\nSuggested optimal number of clusters based on silhouette score: {optimal_n_clusters}")

                # Use the best k-means model
                best_kmeans_model = kmeans_models[optimal_n_clusters]
                peak_labels = best_kmeans_model.predict(peak_coords)

                # Create a label map based on the clustering
                for i, coord in enumerate(peak_coords):
                    peak_cluster_map[tuple(coord)] = peak_labels[i] + 1  # Add 1 to avoid background label 0
                nib.save(nib.Nifti1Image(peak_cluster_map, affine), 'peak_clusters.nii.gz')
            else:
                print("Could not determine optimal number of clusters.")
                peak_cluster_map = markers # Fallback to individual peaks if clustering fails
        else:
            print("No peaks found, cannot perform clustering.")
            peak_cluster_map = markers # Fallback if no peaks are found
    else:
        peak_cluster_map = markers

    print(np.count_nonzero(peak_cluster_map), np.unique(peak_cluster_map))
    # Apply the watershed algorithm with compactness parameter
    labels = watershed(-distance_map, peak_cluster_map, mask=mask_data, compactness=compactness_factor)

    # Save the output
    output_img = nib.Nifti1Image(labels * mask_orig, affine)
    nib.save(output_img, 'labels_refined_peaks.nii.gz')

    return output_img

import sys
def main():
    ref_img = nib.load(sys.argv[1])

    mask_data = nib.load(sys.argv[2]).get_fdata().astype(np.uint8)
    mask_data[mask_data > 0] = 1
    smoothness = 0.0  # Experiment with different values
    compactness = 0.1 # Experiment with different values
    # segmented_image = identify_lesion_blobs_meanshift(ref_img.get_fdata(), mask_data, affine=ref_img.affine)
    segmented_image = segment_lesions(mask_data, ref_img.get_fdata(),
                                      smoothness_factor=smoothness,
                                      compactness_factor=compactness,
                                      affine=ref_img.affine)


if __name__ == '__main__':
    main()