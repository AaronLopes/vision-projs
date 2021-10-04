#!/usr/bin/python3

import numpy as np
import pdb
import torch
import sys
import os
sys.path.append(os.getcwd())
from student_code import IoU


def test_IoU1():
    """"""
    pred = torch.tensor([[1, 0], [1, 0]])
    target = torch.tensor([[1, 0], [1, 1]])
    
    iou = IoU(pred,target)
    
    assert 0.66 <= iou
    assert iou <= 2/3
    
def test_IoU2():
    """"""
    pred = torch.tensor([[1, 0, 0], [1, 0, 0]])
    target = torch.tensor([[1, 0, 1], [1, 1, 0]])
    
    iou = IoU(pred,target)
    
    assert iou == 0.5