import student_code
import numpy as np
from pathlib import Path
from proj5_unit_tests import test_part1
from utils import load_image

# Load the data
"""
points_2d_pic_a = np.loadtxt('../data/CCB_GaTech/pts2d-pic_a.txt')
points_2d_pic_b = np.loadtxt('../data/CCB_GaTech/pts2d-pic_b.txt')
img_a = load_image('../data/CCB_GaTech/pic_a.jpg')
img_b = load_image('../data/CCB_GaTech/pic_b.jpg')
"""

points1 = np.array(
    [
        [886.0, 347.0],
        [943.0, 128.0],
        [476.0, 590.0],
        [419.0, 214.0],
        [783.0, 521.0],
        [235.0, 427.0],
        [665.0, 429.0],
        [525.0, 234.0],
    ],
    dtype=np.float32,
)

points2 = np.array(
    [
        [903.0, 342.0],
        [867.0, 177.0],
        [958.0, 572.0],
        [328.0, 244.0],
        [1064.0, 470.0],
        [480.0, 495.0],
        [964.0, 419.0],
        [465.0, 263.0],
    ],
    dtype=np.float32,
)

F_student = student_code.estimate_fundamental_matrix(points1, points2)
