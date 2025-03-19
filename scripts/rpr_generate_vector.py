
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


def generate_dirac_sh(normal_vectors, sphere, sh_order):
    """Generates SH coefficients approximating a Dirac delta function pointing in the normal direction."""
    # We can approximate a Dirac by projecting the normal vector onto a high-order SH basis.
    # A simple approach is to use the normal vector itself as the direction of a very peaked function.
    # For a Dirac-like SH, we want it to be strongly aligned with the normal.
    # Let's try a simple approach: Project the normal vector onto the SH basis.

    # Generate SH basis for the given order
    dot_products = np.dot(sphere.vertices, normal_vectors.T)
    # Using exponential for a sharper falloff, you can experiment with powers too.
    # e.g., return dot_product ** sharpness_factor  (if sharpness_factor is integer and even)
    # Maximum value when dot_product = 1 (direction = normal_direction)
    SFs = np.exp(4.0 * (dot_products - 1.0))
    SFs /= SFs.max(axis=0)
    SHs = sf_to_sh_internal(SFs.T)
    # SHs = []
    # for SF in SFs.T:
    #     SF /= SF.shape
    #     SH = sf_to_sh(SF, sphere, sh_order_max=sh_order, basis_type="tournier07")
    #     SHs.append(SH)
    return SHs


def generate_donut_sh(normal_vectors, sphere, sh_order):
    """Generates SH coefficients approximating a donut shape perpendicular to the normal direction."""

    # Generate SH basis for the given order
    dot_product = np.dot(sphere.vertices, normal_vectors.T)

    # Spherical Function for Donut: value is low along normal, high perpendicular
    # Using 1 - dot_product**2 to create the donut shape
    # Value will be close to 0 when direction is along normal (dot_product ~ 1 or -1)
    SFs = 1.0 - np.abs(dot_product) ** 0.25
    #  Value will be close to 1 when direction is perpendicular (dot_product ~ 0)

    # Normlize in the axis 0
    SFs /= SFs.max(axis=0)
    # SHs = []
    # for SF in SFs.T:
    #     SF /= SF.max()
    SHs = sf_to_sh_internal(SFs.T)
    #     SHs.append(SH)
    return SHs


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_surface',
                   help='Input file name, in VTK friendly format.')
    p.add_argument('in_fodf',
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

    data_sh = np.zeros(ref_img.shape[0:3] + (28, 3), dtype=float)
    mask_data = np.zeros(ref_img.shape[0:3], dtype=np.uint8)
    mask_data[tuple(valid_coords.T)] = 1
    filename = os.path.join(args.out_dir, "mask.nii.gz")
    nib.save(nib.Nifti1Image(mask_data, ref_img.affine), filename)
    print("Enclosed points time: ", time() - timer)

    # SH_array_sphere = sf_to_sh(np.ones((sphere.vertices.shape[0], 1)))
    distance_map = np.zeros(ref_img.shape, dtype=float)
    for coord in tqdm.tqdm(valid_coords):
        SH_array_sphere = sf_to_sh_internal(
            np.ones((sphere.vertices.shape[0], 1)))
        D, _ = sfs_tree.query(coord, k=1, distance_upper_bound=DISTANCE)
        neighbor_idxs = np.array(sfs_tree.query_ball_point(
            coord, r=min(D + 1, DISTANCE) + 1), dtype=int)

        if len(neighbor_idxs) == 0:
            continue

        # for idx in neighbor_idxs:
        curr_SH = np.zeros((3, 28))
        pts = sfs.vertices[neighbor_idxs]
        normal_vectors = normal_array[neighbor_idxs]
        SH_array_dirac = generate_dirac_sh(normal_vectors, sphere, 6)
        SH_array_donut = generate_donut_sh(normal_vectors, sphere, 6)
        distances = np.linalg.norm(pts - coord, axis=-1) - 0.5
        min_distances = np.min(distances)

        weight_dirac = np.exp((-(distances) ** 2) / ((DISTANCE / 3) ** 2))
        weight_sphere = np.exp(
            (-(min_distances - (DISTANCE / 2)) ** 2) / ((DISTANCE / 3) ** 2))
        weight_donut = np.exp(
            (-(distances - DISTANCE) ** 2) / ((DISTANCE / 3) ** 2))

        curr_SH[0] = np.mean(SH_array_dirac * weight_dirac[:, None], axis=0)
        curr_SH[1] = SH_array_sphere * weight_sphere
        curr_SH[2] = np.mean(SH_array_donut * weight_donut[:, None], axis=0)

        coord = tuple(coord)
        distance_map[coord] += np.min(distances)
        data_sh[coord] += curr_SH.T

    # Compute what is needed for the priors
    indices = np.where(mask_data > 0)
    for i in range(3):
        tmp = data_sh[..., i]
        total_energy = np.max(tmp[indices])
        data_sh[..., i] = np.divide(tmp, total_energy)

    merged_sh = np.sum(data_sh, axis=-1)
    merged_sh[indices] /= np.repeat(np.max(merged_sh[indices],
                                    axis=-1)[:, None], 28, axis=-1)
    todi_obj = TrackOrientationDensityImaging(
        ref_img.shape[0:3], 'repulsion724')
    todi_obj.set_todi_from_sh(merged_sh, mask_data, sh_basis="tournier07")

    input_sh_3d = ref_img.get_fdata(dtype=np.float32)
    input_sh_order = find_order_from_nb_coeff(input_sh_3d.shape)

    # Fancy masking of 1d indices to limit spatial dilation to WM
    sub_mask_3d = np.logical_and(
        mask_data, todi_obj.reshape_to_3d(todi_obj.get_mask()))
    sub_mask_1d = sub_mask_3d.flatten()[todi_obj.get_mask()]
    todi_sf = todi_obj.get_todi()[sub_mask_1d]  # ** 2 # Sharpen

    # The priors should always be between 0 and 1
    # A minimum threshold is set to prevent misaligned FOD from disappearing
    todi_sf /= np.max(todi_sf, axis=-1, keepdims=True)
    todi_sf[todi_sf < args.sf_threshold] = args.sf_threshold

    # Memory friendly saving, as soon as possible saving then delete
    priors_3d = np.zeros(ref_img.shape)
    sphere = get_sphere(name='repulsion724')
    priors_3d[sub_mask_3d] = sf_to_sh(todi_sf, sphere,
                                      sh_order_max=input_sh_order,
                                      basis_type=args.sh_basis,
                                      legacy=True)
    filename = os.path.join(args.out_dir, "priors.nii.gz")
    nib.save(nib.Nifti1Image(priors_3d, ref_img.affine), filename)
    del priors_3d

    input_sf_1d = sh_to_sf(input_sh_3d[sub_mask_3d],
                           sphere, sh_order_max=input_sh_order,
                           basis_type=args.sh_basis, legacy=True)

    # Creation of the enhanced-FOD (direction-wise multiplication)
    mult_sf_1d = input_sf_1d * todi_sf
    del todi_sf

    input_max_value = np.max(input_sf_1d, axis=-1, keepdims=True)
    mult_max_value = np.max(mult_sf_1d, axis=-1, keepdims=True)
    mult_positive_mask = np.squeeze(mult_max_value) > 0.0
    mult_sf_1d[mult_positive_mask] = mult_sf_1d[mult_positive_mask] * \
        input_max_value[mult_positive_mask] / \
        mult_max_value[mult_positive_mask]

    # Memory friendly saving
    input_sh_3d[sub_mask_3d] = sf_to_sh(mult_sf_1d, sphere,
                                        sh_order_max=input_sh_order,
                                        basis_type=args.sh_basis,
                                        legacy=True)
    filename = os.path.join(args.out_dir, "efod.nii.gz")
    nib.save(nib.Nifti1Image(input_sh_3d, ref_img.affine), filename)
    del input_sh_3d

    filename = os.path.join(args.out_dir, "distance_map.nii.gz")
    nib.save(nib.Nifti1Image(distance_map, ref_img.affine), "filename.nii.gz")

    filename = os.path.join(args.out_dir, "data_sh_dirac.nii.gz")
    nib.save(nib.Nifti1Image(data_sh[..., 0], ref_img.affine), filename)

    filename = os.path.join(args.out_dir, "data_sh_sphere.nii.gz")
    nib.save(nib.Nifti1Image(data_sh[..., 1], ref_img.affine), filename)

    filename = os.path.join(args.out_dir, "data_sh_donut.nii.gz")
    nib.save(nib.Nifti1Image(data_sh[..., 2], ref_img.affine), filename)


if __name__ == "__main__":
    main()
