#!/usr/bin/python3

from student_code import apply_mask_to_image
import numpy as np
import pdb
import torch
import sys
import os
sys.path.append(os.getcwd())


def test_final_seg_range():

    image = np.random.randint(0, 255, size=(360, 480, 3))
    mask = np.random.randint(0, 1, size=(360, 480, 1))

    final_seg = apply_mask_to_image(image, mask)

    assert ((final_seg.all() <= 255) and (final_seg.all() >= 0)) == True


def test_final_seg_values():

    image = np.random.randint(0, 255, size=(360, 480, 3))
    mask = np.random.randint(0, 1, size=(360, 480, 1))

    final_seg = apply_mask_to_image(image, mask)

    assert np.allclose(torch.sum((final_seg[:, :, 0] > 255).long()), np.sum(
        (mask[:, :, 0] > 0)), atol=3) == True
