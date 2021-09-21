from numpy import dstack, kaiser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import matplotlib.pyplot as plt
import student_code
from utils import load_image, save_image

image1 = load_image(
    '/Users/AaronLopes/Desktop/cs4476/proj1_release/proj1_code/data/1a_dog.bmp')
image2 = load_image(
    '/Users/AaronLopes/Desktop/cs4476/proj1_release/proj1_code/data/1b_cat.bmp')

m = image1.shape[0]
n = image1.shape[1]
c = image1.shape[2]
print(image1.shape)
wtv = torch.randn(3, 3)
filter_e = torch.rand(3, 3)
print(torch.mul(wtv, filter_e))

filter = student_code.create_2d_gaussian_kernel(standard_deviation=7)
k = filter.shape[0]
j = filter.shape[1]
# print('filter shape: ', filter.shape)
# print('image1 shape: \n', image1.shape)
# print('filter: \n', filter)
filtered_image = []
r_i = 0
c_i = 0


for c_i in range(c):
    fil_chan = []
    padded_channel = F.pad(
        image1[:, :, c_i], (k // 2, k // 2, j // 2, j // 2), 'constant', 0)
    for row in range(padded_channel.shape[0] - k + 1):
        fil_chan_row = []
        for col in range(padded_channel.shape[1] - j + 1):
            val = torch.sum(torch.mul(
                padded_channel[row:row + k, col:col + j], filter))
            fil_chan_row.append(val)
        fil_chan.append(fil_chan_row)
    filtered_image.append(torch.tensor(fil_chan))

filtered_image = torch.stack(
    (filtered_image[0], filtered_image[1], filtered_image[2]), dim=2)

plt.imshow((filtered_image*255).byte())
plt.show()
