import os
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance
from itertools import product, combinations


def read_file(filepath):
    """
    Reads a file, ignoring lines starting with # or anything after the last # on a line.
    Removes empty lines and returns a list of cleaned lines as strings.

    :param filepath: Path to the file to be read.
    :return: List of cleaned lines from the file.
    """
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            # Strip comments and empty lines
            cleaned_line = line.split('#')[0].strip()
            if cleaned_line:
                lines.append(cleaned_line)

    return lines


def compare_files(reference_lines, altered_lines):
    """
    Compares two lists of lines using Levenshtein distance.

    :param reference_lines: List of lines from the reference file.
    :param altered_lines: List of lines from the altered file.
    :return: Mapping of reference to altered lines with their closest match, and total distance.
    """
    total_distance = 0
    text_1 = " ".join(reference_lines)
    text_2 = " ".join(altered_lines)
    total_distance = levenshtein_distance(text_1, text_2)

    return total_distance


def plot_levenshtein_distances(file_paths):
    """
    Plots Levenshtein distances between combinations of files as a box plot.

    :param file_paths: List of file paths to compare.
    """
    file_distances = []
    labels = []

    comb = list(product(file_paths, repeat=2))[1:len(file_paths)]
    # comb = list(combinations(file_paths, 2)) 
    # Generate combinations of files
    for ref_file, alt_file in comb:
        ref_lines = read_file(ref_file)
        alt_lines = read_file(alt_file)

        total_distance = compare_files(ref_lines, alt_lines)
        file_distances.append(total_distance)
        # labels.append(f"{os.path.basename(alt_file)}")
        labels.append(f"{os.path.basename(ref_file)} vs {os.path.basename(alt_file)}")

    # Create the bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(labels, file_distances, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Levenshtein Distance", fontsize=12)
    plt.title("Levenshtein Distances Between File Combinations", fontsize=14)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Compare files using Levenshtein distance.")
    parser.add_argument("files", nargs='+',
                        help="List of files to compare. First is the reference.")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("You must provide at least one reference file and one altered file.")
    else:
        file_paths = args.files

        # Compute pairwise comparisons and plot results
        plot_levenshtein_distances(file_paths)
