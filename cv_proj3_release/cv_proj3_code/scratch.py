import numpy as np
import torch

x = torch.randn(4, 15)
print(x)
print(torch.argmax(x, 1))
