# -*- coding: utf-8 -*-

import numpy as np

from scilpy.tractograms.streamline_operations import (remove_overlapping_points_streamlines,
                                                      remove_single_point_streamlines,
                                                      cut_invalid_streamlines)


def generate_rotation_matrix(angles, translation=None):
    rotation_matrix = np.eye(4)
    x_rot = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
    y_rot = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    z_rot = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
    rotation_matrix[:3, :3] = np.dot(np.dot(x_rot, y_rot), z_rot)
    rotation_matrix[:3, 3] = translation if translation is not None else 0
    return rotation_matrix


def _clean_sft(sft):
    curr_sft, _ = cut_invalid_streamlines(sft)
    curr_sft = remove_single_point_streamlines(curr_sft)
    curr_sft = remove_overlapping_points_streamlines(curr_sft)

    return curr_sft
