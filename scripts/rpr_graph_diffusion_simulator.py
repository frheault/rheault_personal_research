"""
Graph-Based Diffusion Simulation for Tumor Growth Modeling.

This script simulates a diffusion process on a graph representation of a 3D
medical image, such as an MRI scan. It is designed to model processes like
tumor cell infiltration into surrounding brain tissue.

The workflow is as follows:
1.  Load multiple tissue probability maps (CSF, GM, WM) and fiber orientation
    data (peaks).
2.  Construct a weighted 3D grid representing tissue resistance to diffusion.
    White matter is configured to have low resistance, facilitating diffusion.
3.  Build a graph where each voxel is a node and edges connect neighbors.
    Edge weights are derived from the grid and fiber orientations, influencing
    the diffusion direction and speed.
4.  Initialize a "seed" (e.g., a tumor mask) with an initial energy level.
5.  Run an iterative diffusion simulation where energy flows from high-energy
    nodes to their neighbors based on edge weights.
6.  Optionally, visualize the diffusion process in real-time with a slice
    viewer.
7.  Save the final energy distribution as a NIfTI file, which represents the
    probability map of diffusion or infiltration.
"""
import argparse
import os
import numpy as np
import networkx as nx
import nibabel as nib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter


class LiveViewer:
    """
    A Matplotlib viewer to display the diffusion process in real-time.
    Shows a 2D slice of the 3D grid and updates it at each iteration.
    Includes a slider to navigate through the Z-axis.
    """

    def __init__(self, initial_grid, initial_mask, view_slice):
        self.grid_shape = initial_grid.shape
        # If no slice is specified, pick one near the center of the initial
        # mask
        if view_slice is None:
            self.current_slice = int(
                np.mean(np.argwhere(initial_mask), axis=0)[2])
        else:
            self.current_slice = view_slice

        if not 0 <= self.current_slice < self.grid_shape[2]:
            raise ValueError(
                "view_slice must be within the grid's z-dimension.")

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.25)

        self.ax.imshow(initial_grid[:, :, self.current_slice].T,
                       cmap='gray', origin='lower', alpha=0.6)
        self.im = self.ax.imshow(np.zeros(
            self.grid_shape[:2]).T, cmap='hot', origin='lower', vmin=0, vmax=1, alpha=0.6)
        self.fig.colorbar(self.im, ax=self.ax)

        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='Z Slice',
            valmin=0,
            valmax=self.grid_shape[2] - 1,
            valinit=self.current_slice,
            valstep=1
        )
        self.slider.on_changed(self._update_slice)
        self.background_grid = initial_grid
        self.current_diffusion_grid = np.zeros_like(initial_grid)

    def _update_slice(self, val):
        self.current_slice = int(self.slider.val)
        self.ax.images[0].set_data(
            self.background_grid[:, :, self.current_slice].T)
        self.ax.images[1].set_data(
            self.current_diffusion_grid[:, :, self.current_slice].T)
        self.fig.canvas.draw_idle()

    def update(self, iteration, diffusion_grid):
        """Updates the plot with new diffusion data."""
        self.current_diffusion_grid = diffusion_grid
        self.im.set_data(diffusion_grid[:, :, self.current_slice].T)
        self.ax.set_title(f"Iteration: {iteration + 1}")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_final(self):
        """Keeps the final plot window open."""
        self.ax.set_title(f"Final State (Slice {self.current_slice})")
        plt.ioff()
        plt.show()


class GraphDiffusionModel:
    """
    A model for simulating diffusion on a grid represented as a graph.

    Converts a 3D numpy array into a graph where voxels are nodes and
    neighbors are connected by edges. Edge weights model resistance to
    diffusion.
    """

    def __init__(self, grid_data, peaks_data, connectivity=26):
        if grid_data.ndim != 3:
            raise ValueError("Input grid_data must be a 3D numpy array.")
        self.grid_data = grid_data
        self.peaks_data = peaks_data
        self.shape = grid_data.shape
        self.graph = nx.DiGraph()
        self.node_map = {}
        self.connectivity = connectivity
        self._create_graph_from_grid()

    def _get_neighbors(self, coord):
        """Gets valid neighbors for a coordinate based on connectivity."""
        x, y, z = coord
        neighbors = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue

                    # Manhattan distance for connectivity check
                    dist = abs(i) + abs(j) + abs(k)
                    if self.connectivity == 6 and dist > 1:
                        continue
                    if self.connectivity == 18 and dist > 2:
                        continue

                    nx, ny, nz = x + i, y + j, z + k
                    if 0 <= nx < self.shape[0] and 0 <= ny < self.shape[1] and 
                            0 <= nz < self.shape[2]:
                        neighbors.append((nx, ny, nz))
        return neighbors

    def _create_graph_from_grid(self):
        """Converts the 3D grid into a networkx graph."""
        print("Creating graph from grid...")
        node_id_counter = 0
        # Process only non-zero voxels
        coords = np.argwhere(self.grid_data > 0)
        for x, y, z in tqdm(coords, desc="Creating nodes"):
            self.graph.add_node(node_id_counter, coord=(x, y, z))
            self.node_map[(x, y, z)] = node_id_counter
            node_id_counter += 1

        print("Adding edges between neighbors...")
        for node_id, data in tqdm(self.graph.nodes(data=True),
                                   total=self.graph.number_of_nodes(),
                                   desc="Creating edges"):
            coord = data['coord']
            neighbors = self._get_neighbors(coord)
            for neighbor_coord in neighbors:
                if neighbor_coord in self.node_map:
                    neighbor_id = self.node_map[neighbor_coord]
                    self.graph.add_edge(node_id, neighbor_id)

    def set_edge_weights(self, weight_function):
        """Sets the weight for every edge using a provided function."""
        print("Setting edge weights...")
        for u, v, data in tqdm(self.graph.edges(data=True),
                                   total=self.graph.number_of_edges()):
            coord_u = self.graph.nodes[u]['coord']
            coord_v = self.graph.nodes[v]['coord']
            intensity_u = self.grid_data[coord_u]
            intensity_v = self.grid_data[coord_v]
            peaks = self.peaks_data[coord_u]
            node_orientation = np.array(coord_v) - np.array(coord_u)
            weight_uv = weight_function(
                intensity_u, intensity_v, node_orientation, peaks)
            self.graph.add_edge(u, v, weight=max(weight_uv, 1e-6))

    def run_diffusion_optimized(self, initial_mask, initial_energy=1.0,
                                diffusion_rate=0.1, iterations=10,
                                use_viewer=True, view_slice=None,
                                save_tmp_folder=None, affine=None):
        """
        Runs the diffusion simulation using an optimized active set approach.
        """
        if initial_mask.shape != self.shape:
            raise ValueError(
                "Initial mask must have the same shape as the grid.")

        if save_tmp_folder:
            os.makedirs(save_tmp_folder, exist_ok=True)
        save_interval = max(1, iterations // 10)

        viewer = LiveViewer(
            self.grid_data, initial_mask, view_slice) if use_viewer else None

        print("Initializing energy...")
        energy = np.zeros(self.graph.number_of_nodes())
        source_nodes = [self.node_map[tuple(coord)] for coord in np.argwhere(
            initial_mask) if tuple(coord) in self.node_map]
        energy[source_nodes] = initial_energy

        active_nodes = set(source_nodes)
        print(f"Starting with {len(active_nodes)} active nodes.")

        pbar = tqdm(range(iterations), desc="Running diffusion")
        for i in pbar:
            if not active_nodes:
                print("Simulation stabilized early. Stopping.")
                break

            new_energy = np.copy(energy)
            nodes_to_update = set()
            for u in active_nodes:
                nodes_to_update.add(u)
                nodes_to_update.update(self.graph.predecessors(u))
                nodes_to_update.update(self.graph.successors(u))

            next_active_nodes = set()
            for u in nodes_to_update:
                net_flow = 0
                for v in self.graph.predecessors(u):
                    weight = self.graph.edges[v, u].get('weight', 1.0)
                    flow = (energy[v] - energy[u]) / weight
                    net_flow += flow

                delta_energy = diffusion_rate * net_flow
                if abs(delta_energy) > 1e-9:
                    new_energy[u] += delta_energy
                    next_active_nodes.add(u)

            energy = new_energy
            np.clip(energy, 0, 1, out=energy)
            active_nodes = next_active_nodes
            pbar.set_description(f"Active nodes: {len(active_nodes)}")

            if viewer or (save_tmp_folder and (i + 1) % save_interval == 0):
                current_grid = self.convert_energy_to_grid(energy)
                if np.max(current_grid) > 0:
                    current_grid /= np.max(current_grid)
                if viewer:
                    viewer.update(i, current_grid)
                if save_tmp_folder and (i + 1) % save_interval == 0:
                    filename = os.path.join(
                        save_tmp_folder, f"iteration_{i+1}.nii.gz")
                    print(f"Saving temporary file to {filename}")
                    nib.save(nib.Nifti1Image(
                        current_grid.astype(np.float32), affine), filename)

        print("Diffusion complete.")
        if np.max(energy) > 0:
            energy /= np.max(energy)

        final_grid = self.convert_energy_to_grid(energy)
        if viewer:
            viewer.show_final()

        return final_grid

    def convert_energy_to_grid(self, energy):
        """Converts final node energy values back to a 3D grid."""
        result_grid = np.zeros(self.shape, dtype=float)
        for node_id, energy_val in enumerate(energy):
            coord = self.graph.nodes[node_id]['coord']
            result_grid[coord] = energy_val
        return result_grid


def default_weight_function(intensity_u, intensity_v, orientation, peaks):
    """
    Calculates diffusion resistance between two voxels.

    The weight represents the 'cost' of moving from voxel u to v.
    - Resistance is primarily based on the destination voxel's intensity
      (`intensity_v`). In the prepared grid, white matter has low intensity
      (low cost), while other tissues have high intensity (high cost).
    - If the movement direction aligns with a major fiber orientation peak,
      the resistance is halved, promoting diffusion along white matter tracts.
    """
    weight = intensity_v

    # Normalize orientation vector
    norm_orientation = np.linalg.norm(orientation)
    if norm_orientation == 0:
        return weight
    orientation = orientation / norm_orientation

    # Check for alignment with fiber orientation peaks
    aligned_found = False
    for i in range(0, peaks.shape[0], 3):
        peak = peaks[i:i+3]
        if np.linalg.norm(peak) == 0:
            continue
        # Check if orientation is within 30 degrees of a peak direction
        if abs(np.dot(orientation, peak)) > 0.866:  # cos(30 degrees)
            aligned_found = True
            break

    if aligned_found:
        weight *= 0.5  # Lower resistance if aligned with fibers

    return max(weight, 1e-6)


def load_and_prepare_grid(csf_path, gm_path, wm_path, wm_sigma, grid_sigma):
    """Loads tissue maps and prepares the diffusion grid."""
    print("Loading and preparing anatomical grid...")
    csf_img = nib.load(csf_path)
    csf_data = csf_img.get_fdata()

    gm_data = nib.load(gm_path).get_fdata()
    wm_data = nib.load(wm_path).get_fdata()

    # Smooth the white matter map to reduce noise
    wm_data = gaussian_filter(wm_data, sigma=wm_sigma)

    # Create a weighted grid: high values are barriers, low values are
    # pathways. WM is weighted heavily to become the main contributor after
    # inversion.
    grid = csf_data * 0.5 + wm_data * 4.0 + gm_data * 1.0
    grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))

    # Invert contrast: make WM low intensity (easy to cross), others high
    grid[grid > 0.0] = 1.0 - grid[grid > 0.0]
    grid = gaussian_filter(grid, sigma=grid_sigma)

    # Create a brain mask to exclude non-brain tissue
    brain_mask = (csf_data + gm_data + wm_data) > 1e-3
    grid[~brain_mask] = 1.0  # Set non-brain areas to max resistance

    # Save the intermediate inverted contrast grid for inspection
    nib.save(nib.Nifti1Image(grid.astype(
        np.float32), csf_img.affine), 'inverted_contrast.nii.gz')

    return grid, csf_img.affine


def create_rectangular_mask(seed_mask, grid_shape, size):
    """
    Creates a rectangular mask around a bounding box of the seed mask.
    An extra margin of half the given size is added to each side.
    """
    coords = np.argwhere(seed_mask)
    if coords.size == 0:
        raise ValueError("Seed mask is empty.")

    x_min, y_min, z_min = coords.min(axis=0)
    x_max, y_max, z_max = coords.max(axis=0)

    half_size = size // 2
    x_start = max(x_min - half_size, 0)
    x_end = min(x_max + half_size, grid_shape[0])
    y_start = max(y_min - half_size, 0)
    y_end = min(y_max + half_size, grid_shape[1])
    z_start = max(z_min - half_size, 0)
    z_end = min(z_max + half_size, grid_shape[2])

    mask = np.zeros(grid_shape, dtype=bool)
    mask[x_start:x_end, y_start:y_end, z_start:z_end] = True
    return mask


def setup_argparser():
    """Configures and returns the argument parser."""
    parser = argparse.ArgumentParser(
        description="Graph-based diffusion simulation for tumor growth.")

    # Input files
    parser.add_argument('seed_mask',
                        help="Path to the NIfTI file for the initial seed mask (e.g., tumor).")
    parser.add_argument('csf_map',
                        help="Path to the NIfTI file for the CSF probability map.")
    parser.add_argument('gm_map',
                        help="Path to the NIfTI file for the gray matter probability map.")
    parser.add_argument('wm_map',
                        help="Path to the NIfTI file for the white matter probability map.")
    parser.add_argument('peaks',
                        help="Path to the NIfTI file for fiber orientation peaks.")
    parser.add_argument('output',
                        help="Path to save the output diffusion map.")

    # Simulation hyperparameters
    parser.add_argument('--diffusion_rate', type=float, default=0.001,
                        help="Diffusion rate.")
    parser.add_argument('--iterations', type=int, default=100,
                        help="Number of diffusion iterations.")
    parser.add_argument('--connectivity', type=int, choices=[6, 18, 26],
                        default=26,
                        help="Voxel connectivity (6, 18, or 26).")
    parser.add_argument('--mask-size', type=int, default=50,
                        help="Padding size to add around the seed mask's bounding box.")

    # Grid preparation parameters
    parser.add_argument('--wm_sigma', type=float, default=1.0,
                        help="Sigma for Gaussian smoothing of the WM map.")
    parser.add_argument('--grid_sigma', type=float, default=0.5,
                        help="Sigma for Gaussian smoothing of the final grid.")

    # Visualization
    parser.add_argument('--no_viewer', action='store_true', help="Disable the live viewer.")
    parser.add_argument('--view_slice', type=int, default=None,
                        help="Z-slice to display in the viewer (defaults to seed center).")

    # Output
    parser.add_argument('--save_tmp_file', type=str, default=None,
                        help="Path to a folder to save intermediate diffusion grids.")

    return parser


def main(args):
    """Main function to run the diffusion simulation."""
    # 1. Load and prepare the grid
    grid, affine = load_and_prepare_grid(
        args.csf_map, args.gm_map, args.wm_map,
        args.wm_sigma, args.grid_sigma)

    # 2. Load seed mask and peaks data
    seed_mask = nib.load(args.seed_mask).get_fdata().astype(bool)
    peaks_data = nib.load(args.peaks).get_fdata()

    # 3. Create a rectangular processing mask to limit computation
    processing_mask = create_rectangular_mask(
        seed_mask, grid.shape, args.mask_size)
    grid_masked = np.zeros_like(grid)
    grid_masked[processing_mask] = grid[processing_mask]

    # 4. Initialize and configure the model
    model = GraphDiffusionModel(
        grid_masked, peaks_data, connectivity=args.connectivity)
    model.set_edge_weights(weight_function=default_weight_function)

    # 5. Run the diffusion
    diffusion_result = model.run_diffusion_optimized(
        initial_mask=seed_mask,
        diffusion_rate=args.diffusion_rate,
        iterations=args.iterations,
        use_viewer=not args.no_viewer,
        view_slice=args.view_slice,
        save_tmp_folder=args.save_tmp_file,
        affine=affine
    )

    # 6. Save the output
    print(f"Saving result to {args.output}...")
    reconstructed_grid = np.zeros_like(grid)
    reconstructed_grid[processing_mask] = diffusion_result[processing_mask]
    result_img = nib.Nifti1Image(reconstructed_grid, affine)
    result_img.to_filename(args.output)

    print("
Process finished.")
    print(f"Output saved to '{args.output}'")


if __name__ == '__main__':
    parser = setup_argparser()
    args = parser.parse_args()
    main(args)
