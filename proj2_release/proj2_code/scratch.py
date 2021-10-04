import torch

n = 50
m = 50
c = 3

mask = torch.ones((n, m, c))
colors = {'red': torch.Tensor([255, 0, 0])}
broadcast_test = mask * colors['red']
print(broadcast_test)

img = torch.ones((n, m, c))
seg_img = (img + broadcast_test)
print(seg_img.shape)
print(seg_img)
# seg_img = torch.sum(seg_img)
# print(seg_img)
