"""Student code for Project 0: Pytorch tutorial."""
import torch


def vector_transpose(v: torch.Tensor) -> torch.Tensor:
    """
    In this method we will take a row vector and transpose it to be
    a column vector


    Useful functions:
    -  torch.transpose 

    Args:
    -   v: 1 x 3 torch.FloatTensor

    Returns:
    -   v_t: 3 x 1 torch.FloatTensor
    """
    # v_t is the placeholder for the result
    v_t = torch.tensor([[v[0]], [v[1]], [v[2]]])
    return v_t


def stack_images(red_channel_image: torch.Tensor,
                 green_channel_image: torch.Tensor,
                 blue_channel_image: torch.Tensor) -> torch.Tensor:
    """
    In this method we will work on matrix manipulation. 
    This method recieves three gray images X, Y and Z and you are will
    take one layer of each image and stack them to create a new image.
    Suggestion, use the torch.stack function and stack the images in dim = 2.


    Useful functions:
    -   torch.stack()

    Args:
    -   red_channel_image: M x N x 1 torch.FloatTensor
    -   green_channel_image: M x N x 1 torch.FloatTensor
    -   blue_channel_image: M x N x 1 torch.FloatTensor

    Returns:
    -   D: M x N x 3
    """
    # D is the placeholder for the result
    color_image = torch.cat((torch.unsqueeze(red_channel_image, 2), torch.unsqueeze(
        green_channel_image, 2), torch.unsqueeze(blue_channel_image, 2)), 2)
    return color_image


def concat_images(X: torch.Tensor) -> torch.Tensor:
    """
    In this method we will work on matrix manipulation. 
    This method recieves one color image and you will create a 
    2x2 array of that image, such that
     _________
    | Im | Im |
    |____|____|
    | Im | Im |
    |____|____|

    Useful functions:
    -   torch.cat()

    Args:
    -   X: M x N x 3 torch.FloatTensor

    Returns:
    -   D: 2M x 2N x 3 torch.FloatTensor
    """
    # D is the placeholder for the result
    R1 = torch.cat((X, X), 0)
    R2 = torch.cat((X, X), 0)
    D = torch.cat((R1, R2), 1)
    return D


def create_mask(X: torch.Tensor, val: float) -> torch.Tensor:
    """
    In this method you recieve a matrix (X) and will create a new matrix
    of the same size NxM with zeros where the pixel is greater than 
    the set value (val) and a 1 where the pixel is less or equal than the set value

    Useful functions:
    -   torch.le()

    Args:
    -   X: M x N torch.FloatTensor
    -   val: float value in [0, 1] which serves as the threshold for the mask

    Returns:
    -   mask: M x N torch.FloatTensor
    """
    # mask is the placeholder for the result
    mask = torch.le(X, val) * 1
    return mask
