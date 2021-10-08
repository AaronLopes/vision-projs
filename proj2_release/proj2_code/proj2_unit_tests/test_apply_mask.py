#!/usr/bin/python3

import numpy as np
import pdb
import torch
import sys
import os
sys.path.append(os.getcwd())
from student_code import apply_mask_to_image


def test_final_seg_range():
    
    image = np.random.randint(0, 255, size=(360, 480, 3))
    mask = np.random.randint(0, 2, size=(360, 480,1))
    
    final_seg = apply_mask_to_image(image, mask)
    
    assert ( (final_seg <= 255).all() and (final_seg >= 0).all() ) == True 
        
def test_final_seg_values():
    
    image = np.random.randint(0, 255, size=(360, 480, 3))
    mask = np.random.randint(0, 2, size=(360, 480,1))
    
    final_seg = apply_mask_to_image(image, mask)
    
    assert np.allclose(torch.sum((final_seg[:,:,0]==255).long()), np.sum((mask[:,:,0]>0)), atol=3) == True