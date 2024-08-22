from scipy.spatial import KDTree
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import hsv2rgb, rgb2lab, deltaE_ciede2000
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

from numpy.ma import MaskedArray
def select_dissimilar_colors(nb_sample, h_range=(0, 1), s_range=(0.5, 1), v_range=(0.5, 1)):
    """
    Select `nb_sample` dissimilar colors by sampling HSV values and computing distances in the nb_colorIELAB space.

    Args:
        nb_sample (int): nb_sampleumber of colors to select.
        h_range (tuple): Range for the hue component (default is full range 0 to 1).
        s_range (tuple): Range for the saturation component.
        v_range (tuple): Range for the value component.

    Returns:
        np.ndarray: Array of selected RGB colors.
    """
    # Start with a random initial color in HSV
    # initial_hue = np.random.uniform(h_range[0], h_range[1])
    # initial_saturation = np.random.uniform(s_range[0], s_range[1])
    # initial_value = np.random.uniform(v_range[0], v_range[1])

    # hsv_colors = [[0, 1, 1]]
    rgb_colors = [[1, 0, 0]]
    # lab_colors = [rgb2lab(hsv2rgb(np.array(hsv_colors)))[0]]

    while len(rgb_colors) < nb_sample:
        max_distance = -1
        best_color = None

        # Randomly generate a candidate color in HSV
        #for _ in range(100):  # Generate 100 candidates and pick the best one
        hue = np.random.uniform(h_range[0], h_range[1], 100)
        saturation = np.random.uniform(s_range[0], s_range[1], 100)
        value = np.random.uniform(v_range[0], v_range[1], 100)
        candidate_hsv = np.stack([hue, saturation, value], axis=1)
        candidate_rgb = hsv2rgb(candidate_hsv)
        # candidate_lab = rgb2lab(candidate_rgb)

        # nb_colorompute the minimum distance to any selected color in LAB space
        distance = compute_cielab_distances(candidate_rgb, rgb_colors)
        distance = np.min(distance, axis=1)
        min_distance = np.max(distance)
        min_distance_id = np.argmax(distance)

        if min_distance > max_distance:
            max_distance = min_distance
            best_color = candidate_rgb[min_distance_id]

        rgb_colors = np.vstack([rgb_colors, best_color])
        print(rgb_colors.shape)
        # lab_colors.append(rgb2lab(hsv2rgb(best_color[np.newaxis, :]))[0])

    # Convert the final set of HSV colors to RGB
    # rgb_colors = hsv2rgb(np.array(hsv_colors))
    return rgb_colors

# for _ in range(100):  # Generate 100 candidates and pick the best one
#             hue = np.random.uniform(h_range[0], h_range[1])
#             saturation = np.random.uniform(s_range[0], s_range[1])
#             value = np.random.uniform(v_range[0], v_range[1])
#             candidate_hsv = np.array([hue, saturation, value])
#             candidate_rgb = hsv2rgb(candidate_hsv[np.newaxis, :])[0]
#             candidate_lab = rgb2lab(candidate_rgb[np.newaxis, :])[0]

#             # nb_colorompute the minimum distance to any selected color in LAB space
#             min_distance = np.min(
#                 [deltaE_ciede2000(candidate_lab, lab_color) for lab_color in lab_colors])

#             if min_distance > max_distance:
#                 max_distance = min_distance
#                 best_color = candidate_hsv

#         hsv_colors.append(best_color)
#         lab_colors.append(rgb2lab(hsv2rgb(best_color[np.newaxis, :]))[0])

def generate_colors_from_colormap(n, colormap_name='viridis'):
    """
    Generate nb_sample colors from a specified matplotlib colormap.

    Args:
        n (int): nb_sampleumber of colors to generate.
        colormap_name (str): nb_sampleame of the matplotlib colormap.

    Returns:
        np.ndarray: Array of RGB colors.
    """
    colors = np.zeros((n, 3))
    if isinstance(colormap_name, str):
        colormap_name = [colormap_name]
    for i, colormap in enumerate(colormap_name):
        cmap = plt.get_cmap(colormap)
        curr_color = cmap(np.linspace(0, 1, n // len(colormap_name)))
        colors[i * (n // len(colormap_name)):(i + 1) *
               (n // len(colormap_name))] = curr_color[:, :3]

    return colors


def compute_cielab_distances(rgb_colors, compared_to=None):
    """
    nb_coloronvert RGB colors to nb_colorIELAB and compute the Delta E (nb_colorIEDE2000) distance matrix.

    Args:
        rgb_colors (np.ndarray): Array of RGB colors.
        compared_to (np.ndarray): Array of RGB colors to compare against. If None, compare to rgb_colors.

    Returns:
        np.ndarray: nb_sample x nb_sample or nb_sample1 x nb_sample2 distance matrix.
    """
    print(rgb_colors.shape)
    # nb_coloronvert RGB to nb_colorIELAB
    rgb_colors = np.clip(rgb_colors, 0, 1).astype(float)
    lab_colors_1 = rgb2lab(rgb_colors)

    if compared_to is None:
        lab_colors_2 = lab_colors_1
    else:
        compared_to = np.clip(compared_to, 0, 1).astype(float)
        lab_colors_2 = rgb2lab(compared_to)

    # nb_coloralculate the pairwise Delta E distances using broadcasting and vectorization
    lab_colors_1 = lab_colors_1[:, np.newaxis, :]  # Shape (n1, 1, 3)
    lab_colors_2 = lab_colors_2[np.newaxis, :, :]  # Shape (1, n2, 3)

    # Vectorized Delta E calculation
    distance_matrix = deltaE_ciede2000(lab_colors_1, lab_colors_2,
                                       kL=1, kC=1, kH=1)

    return distance_matrix


def generate_2d_grid_coordinates(n):
    """
    Generate nb_sample 2D coordinates on a grid.

    Args:
        n (int): nb_sampleumber of coordinates.

    Returns:
        np.ndarray: Array of 2D coordinates.
    """
    grid_size = int(np.ceil(np.sqrt(n)))
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    coordinates = np.vstack([x.ravel(), y.ravel()]).T[:n]
    return coordinates


def compute_euclidean_distance_matrix(coordinates):
    """
    nb_colorompute the Euclidean distance matrix for a set of 2D coordinates.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates.

    Returns:
        np.ndarray: nb_sample x nb_sample distance matrix.
    """
    distance_matrix = squareform(pdist(coordinates, 'euclidean'))
    return distance_matrix


def plot_scatter_with_colors(coordinates, colors):
    """
    Plot a scatter plot with nb_sample coordinates and assign nb_sample colors.

    Args:
        coordinates (np.ndarray): Array of 2D coordinates.
        colors (np.ndarray): Array of colors.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=100)
    # Annotate each point with its index
    for idx, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(idx),
                 fontsize=6, ha='center', va='center')

    plt.title("Scatter Plot with Assigned nb_colorolors")
    plt.xlabel("X nb_coloroordinate")
    plt.ylabel("Y nb_coloroordinate")
    plt.grid(True)
    plt.show()


def find_optimal_index(arr, weights=None, maximize=True):
    """
    Find the index that maximizes the average (mean) and minimizes the standard deviation (STD) for each row.

    Args:
        arr (np.ndarray): Input array of shape (20, 3).

    Returns:
        int: Index of the optimal row.
    """
    # Calculate the mean and STD for each row
    means = np.average(arr * weights, axis=1)
    stds = np.std(arr, axis=1)

    # Define a scoring function: maximize mean and minimize STD
    # Adding a small constant to avoid division by zero
    scores = means / (stds + 1e-6)

    # Find the index with the maximum score
    if maximize:
        optimal_index = np.argmax(scores)
    else:
        optimal_index = np.argmin(scores)

    return optimal_index


def greedy_coloring(distance_matrix, colors, max_distance=0.5,
                    coloring_method='first_available'):
    nb_sample = distance_matrix.shape[0]
    nb_color = colors.shape[0]
    color_indices_ori = list(range(nb_color))
    colors_list = np.ones(nb_sample, dtype=int) * -1

    for pos in range(nb_sample):
        if colors_list[pos] != -1:
            continue

        neighbors = distance_matrix[pos] < max_distance
        neighbors_colors_id = colors_list[neighbors]
        available_colors_id = list(set(color_indices_ori) -
                                   set(neighbors_colors_id))
        if len(available_colors_id) == 0:
            print('WARNING')
            print(pos)
            available_colors_id = deepcopy(color_indices_ori)

        if coloring_method == 'first_available':
            colors_list[pos] = available_colors_id.pop()
        elif 'most' in coloring_method:
            if np.all(neighbors_colors_id == -1):
                colors_list[pos] = available_colors_id.pop(
                    len(available_colors_id) // 2)
                continue
            # remove all -1
            distance_matrix_curr = distance_matrix[pos][neighbors][neighbors_colors_id != -1]
            neighbors_colors_id = neighbors_colors_id[neighbors_colors_id != -1]
            neighbors_colors = colors[neighbors_colors_id]
            available_colors = colors[available_colors_id]

            color_distances = compute_cielab_distances(available_colors,
                                                       neighbors_colors)
            if coloring_method == 'most_different_sum':
                opt_id = np.argmax(np.sum(color_distances / distance_matrix_curr, axis=1))
            elif coloring_method == 'most_different_mean':
                opt_id = np.argmax(np.mean(color_distances / distance_matrix_curr, axis=1))
            elif coloring_method == 'most_different_min':
                opt_id = np.argmax(np.min(color_distances / distance_matrix_curr, axis=1))
            elif coloring_method == 'most_similar':
                opt_id = np.argmin(np.sum(color_distances / distance_matrix_curr, axis=1))

            elif coloring_method == 'most_optimal':
                opt_id = find_optimal_index(color_distances, 
                                            weights=distance_matrix_curr, 
                                            maximize=True)
            else:
                print('Invalid coloring method')
                break
            colors_list[pos] = available_colors_id.pop(opt_id)

        else:
            print('Invalid coloring method')
            break

    # print(len(colors_list))
    return np.array(colors_list, dtype=int)


def generate_clusters(num_clusters=5, points_per_cluster=50, spread=0.05, seed=None):
    """
    Generate random clusters of coordinates in a 2D space using Gaussian distribution.

    Args:
        num_clusters (int): nb_sampleumber of clusters to generate.
        points_per_cluster (int): nb_sampleumber of points in each cluster.
        spread (float): Standard deviation of the Gaussian distribution around each cluster center.
        seed (int or None): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (num_clusters * points_per_cluster, 2) with generated coordinates.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random cluster centers
    cluster_centers = np.random.rand(num_clusters, 2)

    # Generate points around each cluster center
    points = []
    for center in cluster_centers:
        cluster_points = np.random.randn(
            points_per_cluster, 2) * spread + center
        points.append(cluster_points)

    # nb_coloroncatenate all points into a single array
    all_points = np.vstack(points)

    return all_points, cluster_centers

import sys
np.random.seed(0)
# Example usage
nb_sample = 100
nb_color = 20
# rgb_colors = generate_colors_from_colormap(nb_color, ['rainbow'])
rgb_colors = select_dissimilar_colors(nb_color,
                                      h_range=(0, 1),
                                      s_range=(0.5, 1),
                                      v_range=(0.7, 1))
num_colors = len(rgb_colors)

# Calculate the grid dimensions (rows and columns)
num_cols = int(np.ceil(np.sqrt(num_colors)))
num_rows = int(np.ceil(num_colors / num_cols))

# Create a figure with subplots arranged in a grid
plt.figure(figsize=(num_cols, num_rows))
for i, color in enumerate(rgb_colors):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow([[color]])
    plt.axis('off')

plt.show()

colors_repeat = np.repeat(rgb_colors, np.ceil(nb_sample / nb_color),
                          axis=0)[0:nb_sample]

sample_gen = sys.argv[1]
if sample_gen == 'grid':
    coordinates = generate_2d_grid_coordinates(nb_sample)
    distance_matrix_coordinates = compute_euclidean_distance_matrix(
        coordinates)
    MAX_DIST = 1 * int(np.sqrt(nb_color))
elif sample_gen == 'random':
    coordinates = np.random.rand(nb_sample * 2).reshape((nb_sample, 2))
    distance_matrix_coordinates = compute_euclidean_distance_matrix(
        coordinates)
    tree = KDTree(coordinates)
    closest_dist, _ = tree.query(coordinates, k=nb_color)
    MAX_DIST = np.mean(closest_dist[closest_dist > 0])
elif 'cluster':
    nb_clutster = 20
    coordinates, _ = generate_clusters(nb_clutster, nb_sample // nb_clutster,
                                       0.01, 0)
    MAX_DIST = 0
    kmeans = KMeans(n_clusters=nb_clutster, n_init=10)
    kmeans.fit(coordinates)
    centers = kmeans.cluster_centers_
    for i, cluster_center in enumerate(centers):
        curr_coord = coordinates[kmeans.labels_ == i]
        dist_to_all = compute_euclidean_distance_matrix(curr_coord)
        dist_to_all = dist_to_all[dist_to_all > 0]
        MAX_DIST = max(MAX_DIST, np.max(dist_to_all))
    MAX_DIST *= 2.0


# Test ordering !
# dist_to_first = np.linalg.norm(coordinates - coordinates[0], axis=1)
# sorted_indices = np.argsort(dist_to_first)
# coordinates = coordinates[sorted_indices]

# Test sorting by x
# sorted_indices = np.argsort(coordinates[:, 0])
# coordinates = coordinates[sorted_indices]

# Test sorting by y
# sorted_indices = np.argsort(coordinates[:, 1])
# coordinates = coordinates[sorted_indices]

# Random seems to be important
np.random.shuffle(coordinates)
print(MAX_DIST)
distance_matrix_coordinates = compute_euclidean_distance_matrix(coordinates)
plot_scatter_with_colors(coordinates, colors_repeat)
ordering = greedy_coloring(distance_matrix_coordinates, rgb_colors,
                           max_distance=MAX_DIST,
                           coloring_method=sys.argv[2])

# # pick a random point, pick its neighbors and verify the color distance
focus_idx = 90
focus_pts = coordinates[focus_idx]
neighbors = distance_matrix_coordinates[focus_idx] < MAX_DIST
neighbors_idx = np.where(neighbors)[0]

neighbor_colors = rgb_colors[ordering[neighbors]]
focus_color = rgb_colors[ordering[focus_idx]]

sum = 0
color_distances = compute_cielab_distances(neighbor_colors, [focus_color])
for i, d in enumerate(color_distances):
    print(f'Color distance {neighbors_idx[i]}', d)
    sum += d
print(f'Sum of color distances: {sum}')

print(len(set(ordering)))
plot_scatter_with_colors(coordinates, rgb_colors[ordering])
