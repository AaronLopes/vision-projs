import torch
from torch import nn
import student_code

R = torch.randn((9, 9))
median = torch.median(R)
R[R < median] = 0
R = R[None, None, :]
m = torch.nn.MaxPool2d(kernel_size=9)
maxpooled_R = m(R)
print(maxpooled_R[0][0])
