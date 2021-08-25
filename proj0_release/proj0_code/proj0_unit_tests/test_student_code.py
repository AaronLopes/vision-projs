
import os
import unittest

import torch
from proj0_code import student_code
import proj0_code.utils as proj0_utils


def verify(function) -> str:
    """ Will indicate with a print statement whether assertions passed or failed
      within function argument call.

      Args:
      - function: Python function object

      Returns:
      - string
    """
    try:
        function()
        return "\x1b[32m\"Correct\"\x1b[0m"
    except AssertionError:
        return "\x1b[31m\"Wrong\"\x1b[0m"

def resolve_image_path(image_name: str) -> str:
    return os.path.join('imgs', image_name)


def test_vector_transpose():
    """
    Testing vector_transpose()
    """
    v1 = torch.tensor([1., 2., -3.])
    val = torch.tensor([[1.], [2.], [-3.]])
    v_t = student_code.vector_transpose(v1)
    
    assert torch.all(val.eq(v_t)) == True


def test_stack_images():
    """
    Testing stack_images()
    """
    images = [] 
    images.append(proj0_utils.load_image(
        resolve_image_path('fish.jpeg')))
    images.append(proj0_utils.load_image(
        resolve_image_path('fish_R.png')))
    images.append(proj0_utils.load_image(
        resolve_image_path('fish_G.png')))
    images.append(proj0_utils.load_image(
        resolve_image_path('fish_B.png')))

    D = student_code.stack_images(images[1][:,:,0], images[2][:,:,0], images[3][:,:,0])
    
    assert torch.all(D.eq(images[0])) == True

def test_concat_images():
    """
    Testing concat_images()
    """
    images = [] 
    images.append(proj0_utils.load_image(
        resolve_image_path('fish.jpeg')))
    images.append(proj0_utils.load_image(
        resolve_image_path('fish_R.png')))
    images.append(proj0_utils.load_image(
        resolve_image_path('fish_G.png')))
    images.append(proj0_utils.load_image(
        resolve_image_path('fish_B.png')))
    images.append(proj0_utils.load_image(
        resolve_image_path('4fish.png')))

    D = student_code.concat_images(images[0])
    D_test = images[4]

    assert torch.all(D.eq(D_test)) == True

def test_create_mask():
    """
    Testing concat_images()
    """
    
    original = proj0_utils.load_image(resolve_image_path('fish.jpeg'))
    test_mask = proj0_utils.load_image(resolve_image_path('mask.png'))

    mask = student_code.create_mask(original[:,:,0], 0.04)
    aux = mask.type(torch.FloatTensor)

    assert torch.all(aux.eq(test_mask[:,:,0])) == True

