import torch
import numpy as np
from torch import max_pool2d, nn
import student_code

R = torch.from_numpy(np.array(
    [
        [1, 2, 2, 1, 2],
        [1, 6, 2, 1, 1],
        [2, 2, 1, 1, 1],
        [1, 1, 1, 7, 1],
        [1, 1, 1, 1, 1]
    ]).astype(np.float32))
a = torch.tensor([[[[1.,  2,  3,  4],
                    [5,  6,  7,  8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]]]])
median = torch.median(R)
R[R < median] = 0
pool = nn.MaxPool2d(3, 1, padding=1)
R = R[None, None, :]
pooled_R = pool(R)
pooled_R = pooled_R[0][0]
binary_R = []
for r in range(pooled_R.shape[0]):
    bin_row = []
    for c in range(pooled_R.shape[1]):
        if pooled_R[r][c] == R[0][0][r][c]:
            bin_row.append(1)
        else:
            bin_row.append(0)
    binary_R.append(bin_row)
binary_R = torch.tensor(binary_R)
masked_R = R * binary_R
print(masked_R)
k = 2
o, i = torch.topk(masked_R[0][0].flatten(), 2)
indices = np.array(np.unravel_index(i.numpy(), masked_R[0][0].shape)).T
x = torch.tensor(indices[:, 0])
print(o)
