
import argparse
import os
import shutil
from time import time
import tqdm

from dipy.reconst.shm import (real_sh_descoteaux, real_sh_tournier,
                              smooth_pinv, sh_to_sf, sf_to_sh)
from dipy.data import get_sphere
from dipy.io.surface import load_surface
from dipy.io.utils import Space
import nibabel as nib
import numpy as np
from scipy.spatial import KDTree
import vtk
from vtk.util.numpy_support import vtk_to_numpy


from scilpy.io.utils import (add_overwrite_arg,
                             add_reference_arg,
                             add_sh_basis_args,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             parse_sh_basis_arg,
                             assert_headers_compatible)
from scilpy.reconst.utils import find_order_from_nb_coeff
from scilpy.tractanalysis.todi import TrackOrientationDensityImaging


def create_binary_mask_from_surface(surface_polydata, volume_shape, volume_spacing, volume_origin):
    """
    Generates a binary NumPy array mask from a VTK PolyData surface.

    Args:
        surface_polydata (vtk.vtkPolyData): The input surface as a VTK PolyData object.
        volume_shape (tuple): Shape of the volume (x, y, z).
        volume_spacing (tuple): Spacing of voxels in each dimension (x, y, z).
        volume_origin (tuple): Origin of the volume in world coordinates (x, y, z).

    Returns:
        numpy.ndarray: A binary NumPy array (dtype=np.uint8) where 1 represents voxels inside the surface
                       and 0 represents voxels outside.
    """

    # 1. Calculate Normals (important for inside/outside determination)
    polydata_normals = vtk.vtkPolyDataNormals()
    polydata_normals.AutoOrientNormalsOn()
    polydata_normals.SetInputData(surface_polydata)
    polydata_normals.Update()

    # 2. PolyData to Stencil
    polydata_to_stencil = vtk.vtkPolyDataToImageStencil()
    polydata_to_stencil.SetOutputSpacing(volume_spacing)
    polydata_to_stencil.SetOutputOrigin(volume_origin)
    polydata_to_stencil.SetOutputWholeExtent(
        0, volume_shape[0]-1, 0, volume_shape[1]-1, 0, volume_shape[2]-1)  # Extent is 0 to size-1
    polydata_to_stencil.SetInputConnection(polydata_normals.GetOutputPort())
    polydata_to_stencil.Update()

    # 3. Stencil to Image (Binary Mask)
    stencil_to_image = vtk.vtkImageStencilToImage()
    stencil_to_image.SetInputConnection(polydata_to_stencil.GetOutputPort())
    stencil_to_image.SetInsideValue(1)  # 1 for voxels inside the surface
    stencil_to_image.SetOutsideValue(0)  # 0 for voxels outside
    stencil_to_image.Update()

    # 4. Get the resulting vtkImageData as a NumPy array
    vtk_image_data = stencil_to_image.GetOutput()
    mask_array = vtk_to_numpy(vtk_image_data.GetPointData().GetScalars()).reshape(
        volume_shape[::-1]).transpose(2, 1, 0)  # Reshape and transpose

    # Return as binary (0 and 1) numpy array
    return mask_array.astype(np.uint8)


sph_harm_lookup = {None: real_sh_descoteaux,
                   "tournier07": real_sh_tournier,
                   "descoteaux07": real_sh_descoteaux}
sphere = get_sphere(name='symmetric724')
sph_harm_basis = sph_harm_lookup.get("tournier07")
B, m_values, l_values = sph_harm_basis(6, sphere.theta,
                                       sphere.phi,
                                       full_basis=False,
                                       legacy=True)
L = -l_values * (l_values + 1)
invB = smooth_pinv(B, np.sqrt(0.0) * L)


def sf_to_sh_internal(sf):

    if sf.shape[-1] == 1:
        sh = np.dot(invB, sf).T
    else:
        sh = np.dot(sf, invB.T)

    return sh


import heapq
import trimesh
def split_fodf(sphere, sf, peaks):
    mesh = trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces)
    adjacency_list = mesh.vertex_neighbors
    num_vertices = len(mesh.vertices)
    negative_potential = sf
    num_peaks = 3
    
    seed_indices = []
    for i in range(num_peaks):
        peak = peaks[i*3:(i+1)*3]
        if not peak.any():
            continue
        indice = np.argmax(np.dot(sphere.vertices, peak))
        seed_indices.append(indice)

        labels = np.zeros(num_vertices, dtype=int)
        
        # Priority queue stores tuples: (potential_value, vertex_index)
        pq = []

        # Initialize labels and priority queue with seeds
        for i, seed_idx in enumerate(seed_indices):
            seed_label = i + 1 # Assign unique positive labels (1, 2, ...)
            labels[seed_idx] = seed_label
            heapq.heappush(pq, (negative_potential[seed_idx], seed_idx))

        # Process vertices based on potential (lowest first)
            processed_count = 0
            while pq and processed_count < num_vertices:
                current_potential, current_idx = heapq.heappop(pq)
                
                # Check if already processed with a potentially lower path
                # (This check might need refinement depending on exact watershed definition)
                # A simple check is just based on whether it already has a valid label
                if labels[current_idx] == 0 and current_idx not in seed_indices:
                    # This should ideally not happen if seeds are the start points.
                    # Skip or handle as needed. For seeded watershed, we mostly care about neighbors.
                    continue 

                current_label = labels[current_idx]
                if current_label == 0: # Should only happen for non-seeds added later
                    # This logic path needs careful consideration in seeded watershed
                    # Often, items are only added to PQ if neighbours of labelled regions.
                    # Let's adjust the logic: only neighbors of labeled regions get added.
                    continue # Skip if popped vertex wasn't properly labeled by a neighbor yet

                processed_count += 1

                # Examine neighbors
                neighbors = adjacency_list[current_idx]
                for neighbor_idx in neighbors:
                    if labels[neighbor_idx] == 0: # If neighbor is unlabeled
                        labels[neighbor_idx] = current_label # Assign current label
                        heapq.heappush(pq, (negative_potential[neighbor_idx], neighbor_idx))
                    # Optional: Handle boundary condition (watershed lines)
                    # If neighbor has a *different* positive label, mark as boundary?
                    # elif labels[neighbor_idx] != current_label and labels[neighbor_idx] > 0:
                    #     labels[neighbor_idx] = -1 # Mark as boundary (or handle differently)
                        
            # Refined approach: Add neighbours of seeds initially, then expand
            # Let's refine the initialization and loop:

            labels = np.zeros(num_vertices, dtype=int)
            pq = [] # (potential, vertex_idx, label_from)

            # Initialize: Label seeds and add their *neighbors* to the queue
            for i, seed_idx in enumerate(seed_indices):
                seed_label = i + 1
                labels[seed_idx] = seed_label
                # Add neighbors of this seed to the queue
                for neighbor_idx in adjacency_list[seed_idx]:
                    if labels[neighbor_idx] == 0: # Add only if not already labeled by another seed
                        # Add with the potential of the neighbor, associated with this seed's label
                        heapq.heappush(pq, (negative_potential[neighbor_idx], neighbor_idx, seed_label))
                        # Tentatively label to avoid adding multiple times? Or handle conflicts later.
                        # Using a 'visited' or tentative label might be useful here.

            # Process queue (vertices sorted by potential)
            while pq:
                current_potential, current_idx, label_from = heapq.heappop(pq)

                if labels[current_idx] != 0: # Already assigned a final label
                    continue

                # Assign the label from the seed region it's being added from
                labels[current_idx] = label_from

                # Add its unlabeled neighbors to the queue
                for neighbor_idx in adjacency_list[current_idx]:
                    if labels[neighbor_idx] == 0:
                        # Check if already in queue with higher potential? heapq doesn't support decrease-key easily.
                        # Simplest: just add. The first time it's popped with the lowest potential wins.
                        heapq.heappush(pq, (negative_potential[neighbor_idx], neighbor_idx, label_from))
                    # Optional: Detect boundaries
                    # elif labels[neighbor_idx] > 0 and labels[neighbor_idx] != label_from:
                    #     # This vertex is adjacent to two different regions. Mark as boundary?
                    #     labels[current_idx] = -1 # Mark current vertex as boundary
                    #     # Or mark the edge, or handle boundaries differently.

        return labels

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_surface',
                   help='Input file name, in VTK friendly format.')
    p.add_argument('in_fodf',
                   help='Input file name, in nifti format.')
    p.add_argument('in_peaks',
                   help='Input file name, in nifti format.')
    p.add_argument('distance', type=int,
                   help='Distance to consider the neighbors.')
    p.add_argument('out_dir',
                   help='Output directory.')
    p.add_argument('--sf_threshold', type=float, default=0.2,
                   help='Threshold for the SF.')
    add_sh_basis_args(p)
    p.add_argument('--todi_sigma', choices=[0, 1, 2, 3, 4],
                   default=1, type=int,
                   help='Smooth the orientation histogram.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    args.sh_basis = parse_sh_basis_arg(args)[0]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    else:
        shutil.rmtree(args.out_dir)
        os.makedirs(args.out_dir)

    sfs = load_surface(args.in_surface, reference=args.in_fodf,
                       from_space=Space.LPSMM)
    sfs.to_vox()
    sfs.to_center()

    # Generate ranges for each dimension
    DISTANCE = args.distance
    ref_img = nib.load(args.in_fodf)

    timer = time()
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(sfs.get_polydata())  # Set input data

    # Tell it to compute vertex normals (and not cell/face normals, though it might compute both by default)
    # Explicitly ensure vertex normals are computed
    normals_filter.ComputePointNormalsOn()
    # Turn off cell normals if you only need vertex normals (optional, might improve performance)
    normals_filter.ComputeCellNormalsOff()
    # Ensure normals are consistently oriented (e.g., outwards for closed surfaces)
    normals_filter.AutoOrientNormalsOn()
    normals_filter.SplittingOff()  # Usually want splitting off for vertex normals

    # Get the output PolyData with normals
    normals_filter.Update()
    polydata_with_normals = normals_filter.GetOutput()

    # Get the vertex normals as a numpy array
    vertex_normals_vtk_array = polydata_with_normals.GetPointData().GetNormals()
    normal_array = vtk_to_numpy(vertex_normals_vtk_array)
    print("Normal array time: ", time() - timer)

    timer = time()
    sphere = get_sphere(name='symmetric724')
    mask = create_binary_mask_from_surface(
        sfs.get_polydata(), ref_img.shape[0:3], [1, 1, 1], [0, 0, 0])
    print("Mask time: ", time() - timer)

    timer = time()
    enclosed_coords = np.argwhere(mask > 0).astype(np.uint32)
    sfs_tree = KDTree(sfs.vertices)
    has_neighbors = sfs_tree.query_ball_point(
        enclosed_coords, r=DISTANCE, return_length=True)

    has_neighbors = np.where(has_neighbors > 0)[0]
    valid_coords = enclosed_coords[has_neighbors]

    # data_sh = np.zeros(ref_img.shape[0:3] + (28, 3), dtype=float)
    mask_data = np.zeros(ref_img.shape[0:3], dtype=np.uint8)
    mask_data[tuple(valid_coords.T)] = 1
    filename = os.path.join(args.out_dir, "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask_data, ref_img.affine), filename)
    print("Enclosed points time: ", time() - timer)

    peaks_img = nib.load(args.in_peaks)
    peaks_data = peaks_img.get_fdata(dtype=np.float32)

    fodf_img = nib.load(args.in_fodf)
    fodf_data = fodf_img.get_fdata(dtype=np.float32)

    sf_data = sh_to_sf(
        fodf_data[tuple(valid_coords.T)], sphere, sh_order_max=8,
        basis_type=args.sh_basis, legacy=True)
    peaks_data = peaks_data[tuple(valid_coords.T)]

    # SH_array_sphere = sf_to_sh(np.ones((sphere.vertices.shape[0], 1)))
    distance_map = np.zeros(ref_img.shape, dtype=float)
    for i, _ in tqdm.tqdm(enumerate(valid_coords), total=len(valid_coords)):
        labels = split_fodf(sphere, sf_data[i], peaks_data[i])
        print(np.unique(labels, return_counts=True))


    filename = os.path.join(args.out_dir, "efod.nii.gz")
    nib.save(nib.Nifti1Image(input_sh_3d, ref_img.affine), filename)
    del input_sh_3d

    filename = os.path.join(args.out_dir, "distance_map.nii.gz")
    nib.save(nib.Nifti1Image(distance_map, ref_img.affine), filename)



if __name__ == "__main__":
    main()
