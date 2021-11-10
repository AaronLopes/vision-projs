import torch
import numpy as np
from torch import max_pool2d, nn
import student_code

window_magnitudes = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0,
         2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
         1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
)

A = 1/8 * np.pi  # squarely in bin [0, pi/4]
B = 3/8 * np.pi  # squarely in bin [pi/4, pi/2]

window_orientations = np.array(
    [
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [A, A, A, A, A, A, A, A, A, A, A, A, A, A, A, A],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
    ]
)

window_magnitudes = torch.from_numpy(window_magnitudes)
window_orientations = torch.from_numpy(window_orientations)

hist = np.histogram()

print(window_magnitudes.shape)
print(window_orientations.shape)

print(window_magnitudes[:4, :4])
print(window_orientations[:4, :4])
