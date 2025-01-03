#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""

import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_fill_holes


def gaussian_blur(data, mask, blur=0.5):
    # normalized convolution of image with mask
    filtr = gaussian_filter(data * mask, sigma=blur)
    weights = gaussian_filter(mask, sigma=blur)
    filtr /= (weights + 1e-8)
    # after normalized convolution, you can choose to delete any data outside the mask:
    filtr *= mask

    return filtr


# Example usage:
sigma_value = 1.0
# kernel = gaussian_kernel_3d(kernel_size, sigma_value)

img = nib.load(sys.argv[1])
ori_data = img.get_fdata().astype(int)

mask = np.zeros(ori_data.shape, dtype=np.float32)
mask[ori_data > 0] = 1

data = np.zeros(ori_data.shape, dtype=float)
for i in np.unique(ori_data)[1:]:
    tmp = np.zeros(ori_data.shape, dtype=np.uint16)
    tmp[ori_data == i] = 1
    tmp = binary_fill_holes(tmp)
    data[tmp > 0] = i

data /= data.max()
results = gaussian_blur(data.astype(np.float32), mask.astype(np.float32))

results = results.astype(np.float32)
nib.save(nib.Nifti1Image(results, img.affine), sys.argv[2])
