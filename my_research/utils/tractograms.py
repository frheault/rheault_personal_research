# -*- coding: utf-8 -*-
import logging

from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
from scilpy.tractanalysis.reproducibility_measures import compute_dice_voxel
from scilpy.tractograms.streamline_operations import (resample_streamlines_step_size,
                                                      compress_sft)
from scilpy.tractograms.streamline_and_mask_operations import cut_outside_of_mask_streamlines
from scilpy.tractograms.tractogram_operations import upsample_tractogram, shuffle_streamlines
from scilpy.tractanalysis.bundle_operations import uniformize_bundle_sft

from dipy.io.stateful_tractogram import StatefulTractogram, set_sft_logger_level
from dipy.tracking.streamline import transform_streamlines
import numpy as np

from my_research.utils.util import generate_rotation_matrix, _clean_sft

set_sft_logger_level(logging.ERROR)


def subsample_streamlines_alter(sft, min_dice=0.90, epsilon=0.01,
                                baseline_sft=None):
    sft.to_vox()
    sft.to_corner()
    if baseline_sft is None:
        original_density_map = compute_tract_counts_map(sft.streamlines,
                                                        sft.dimensions)
    else:
        baseline_sft.to_vox()
        baseline_sft.to_corner()
        original_density_map = compute_tract_counts_map(baseline_sft.streamlines,
                                                        sft.dimensions)
    dice = 1.0
    init_pick_min = 0
    init_pick_max = len(sft)
    previous_to_pick = None
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        to_pick = init_pick_min + (init_pick_max - init_pick_min) // 2
        if to_pick == previous_to_pick:
            logging.warning('No more streamlines to pick, not converging.')
            break
        previous_to_pick = to_pick

        indices = np.random.choice(len(sft), to_pick, replace=False)
        streamlines = sft.streamlines[indices]
        curr_density_map = compute_tract_counts_map(streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Subsampled {to_pick} streamlines, dice: {dice}')

        if dice < min_dice:
            init_pick_min = to_pick
        else:
            init_pick_max = to_pick

    return StatefulTractogram.from_sft(streamlines, sft)


def cut_streamlines_alter(sft, min_dice=0.90, epsilon=0.01, from_start=True):
    sft = resample_streamlines_step_size(sft, 0.5)
    uniformize_bundle_sft(sft, swap=not from_start)
    sft.to_vox()
    sft.to_corner()
    original_density_map = compute_tract_counts_map(sft.streamlines,
                                                    sft.dimensions)
    dice = 1.0
    init_cut_min = 0
    init_cut_max = 1.0
    previous_to_pick = None
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        to_pick = init_cut_min + (init_cut_max - init_cut_min) / 2
        if to_pick == previous_to_pick:
            logging.warning('No more points to pick, not converging.')
            break
        previous_to_pick = to_pick

        streamlines = []
        for streamline in sft.streamlines:
            pos_to_pick = int(len(streamline) * to_pick)
            streamline = streamline[:pos_to_pick]
            streamlines.append(streamline)
        curr_density_map = compute_tract_counts_map(streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Cut {to_pick}% of the streamlines, dice: {dice}')

        if dice < min_dice:
            init_cut_min = to_pick
        else:
            init_cut_max = to_pick
    new_sft = StatefulTractogram.from_sft(streamlines, sft)
    return compress_sft(new_sft)


def upsample_streamlines_alter(sft, min_dice=0.90, epsilon=0.01):
    logging.debug('Upsampling the streamlines by a factor 2x to then '
                  'downsample.')
    upsampled_sft = upsample_tractogram(sft, len(sft) * 2, point_wise_std=0.5,
                                        tube_radius=1.0, gaussian=None,
                                        error_rate=0.1, seed=1234)
    return subsample_streamlines_alter(upsampled_sft, min_dice, epsilon,
                                       baseline_sft=sft)


def trim_streamlines_alter(sft, min_dice=0.90, epsilon=0.01):
    sft.to_vox()
    sft.to_corner()
    original_density_map = compute_tract_counts_map(sft.streamlines,
                                                    sft.dimensions)
    thr_density = 1
    voxels_to_remove = np.where(original_density_map == thr_density)

    dice = 1.0
    init_trim_min = 0
    init_trim_max = np.count_nonzero(voxels_to_remove[0])
    previous_to_pick = None
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        to_pick = init_trim_min + (init_trim_max - init_trim_min) // 2
        if to_pick == previous_to_pick:
            # If too few streamlines are picked, increase the threshold
            # and reinitialize the picking
            if dice > min_dice and thr_density < 5:
                logging.debug('Increasing threshold density to',
                              thr_density + 1)
                thr_density += 1
                voxels_to_remove = np.where(
                    original_density_map == thr_density)
                init_trim_min = 0
                init_trim_max = np.count_nonzero(voxels_to_remove[0])
                dice = 1.0
                previous_to_pick = None
                continue
            else:
                logging.warning('No more voxels to pick, not converging.')
                break
        previous_to_pick = to_pick

        voxel_to_remove = np.where(original_density_map == thr_density)
        indices = np.random.choice(np.count_nonzero(voxel_to_remove[0]),
                                   to_pick, replace=False)
        voxel_to_remove = tuple(np.array(voxel_to_remove).T[indices].T)
        mask = original_density_map.copy()
        mask[voxel_to_remove] = 0

        new_sft = cut_outside_of_mask_streamlines(sft, mask, min_len=10)

        curr_density_map = compute_tract_counts_map(new_sft.streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Trimmed {to_pick} voxels at density {thr_density}, '
                      f'dice: {dice}')

        if dice < min_dice:
            init_trim_max = to_pick
        else:
            init_trim_min = to_pick

    return new_sft


def transform_streamlines_alter(sft, min_dice=0.90, epsilon=0.01):
    sft.to_vox()
    sft.to_corner()
    original_density_map = compute_tract_counts_map(sft.streamlines,
                                                    sft.dimensions)
    dice = 1.0
    angle_min = [0.0, 0.0, 0.0]
    angle_max = [0.1, 0.1, 0.1]
    previous_dice = None
    last_pick = np.array([0.0, 0.0, 0.0])
    rand_val = np.random.rand(3) * angle_max[0]
    axis_choices = np.random.choice(3, 3, replace=False)
    axis = 0
    while dice > min_dice or np.abs(dice - min_dice) > epsilon:
        init_angle_min = angle_min[axis]
        init_angle_max = angle_max[axis]
        to_pick = init_angle_min + (init_angle_max - init_angle_min) / 2

        # Generate a 4x4 matrix from random euler angles
        rand_val = np.array(angle_max)
        rand_val[axis] = to_pick

        angles = rand_val * 2 * np.pi
        rot_mat = generate_rotation_matrix(angles)
        streamlines = transform_streamlines(sft.streamlines, rot_mat)

        curr_sft = StatefulTractogram.from_sft(streamlines, sft)
        curr_sft = _clean_sft(curr_sft)
        curr_density_map = compute_tract_counts_map(curr_sft.streamlines,
                                                    sft.dimensions)
        dice, _ = compute_dice_voxel(original_density_map, curr_density_map)
        logging.debug(f'Transformed {to_pick*360} degree on axis {axis}, '
                      f'dice: {dice}')
        last_pick[axis] = to_pick

        if dice < min_dice:
            angle_max[axis] = to_pick
        else:
            angle_min[axis] = to_pick

        if (previous_dice is not None) \
                and np.abs(dice - previous_dice) < epsilon / 2:
            logging.debug('Not converging, switching axis.\n')
            axis_choices = np.roll(axis_choices, 1)
            axis = axis_choices[0]
        previous_dice = dice

    logging.debug(f'\nFinal angles: {last_pick*360} at dice: {dice}')
    return curr_sft
