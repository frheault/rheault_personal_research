#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def display_histogram(data, log=False, nb_bins=20):
    """
    This function displays a histogram of a given dataset.

    Parameters
    ----------
    data: array-like
        Data to be plotted in the histogram.
    log: boolean, optional
        If true, the data is log-transformed before plotting.
    nb_bins: int, optional
        Number of bins for the histogram plot.

    Returns
    -------
    None
    """
    data = data.ravel() if not log else np.log(data.ravel(),
                                               where=data.ravel() > 0)

    plt.hist(data, bins=nb_bins, color='c')
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Data")
    plt.show()
