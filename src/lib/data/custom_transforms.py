"""
Custom transforms and augmentations
"""

import numpy as np
import torch
import cv2


def pad_mnist(img):
    """
    Padding a batch of (Fashion-)MNIST images from [B,C,28,28] to [B,C,32,32]
    """

    tensor = torch.tensor((), dtype=torch.double)
    img_padded = -1.0 * tensor.new_ones(img.shape[0], img.shape[1], 32, 32)
    img_padded[:, :, 2:30,2:30] = img
    img_padded = img_padded.type(torch.FloatTensor)

    return img_padded


def pad_img(img, target_shape=(32,32)):
    """
    Padding images so that they end un having the desired shape
    If shape is (32,32) and img are MNIST images, it is equivalent to calling pad_mnist()
    """

    if(len(img.shape)==3):
        img = img.view(img.shape[0], 1, img.shape[1], img.shape[2])
    if(img.shape[-2:] == target_shape):
        return img

    img_padded = -1*torch.ones(size=(img.shape[0], img.shape[1], *target_shape))

    idx_left = (target_shape[1]-img.shape[-1])//2
    idx_right = (target_shape[1]-img.shape[-1])//2
    idx_top = (target_shape[0]-img.shape[-2])//2
    idx_bottom = (target_shape[0]-img.shape[-2])//2

    img_padded[:,:, idx_top:-idx_bottom, idx_left:-idx_right] = img

    return img_padded


def downscale_img(img, target_shape=(32,32)):
    """
    Downsscaling an image to fit the desired shape
    """

    if(torch.is_tensor(img)):
        img = img.numpy()
    shape_y, shape_x = img.shape[-2], img.shape[-1]
    fy, fx = target_shape[0] / shape_y,  target_shape[1] / shape_x
    downscaled_img = cv2.resize(img, None, fx=fx, fy=fy,
                                interpolation = cv2.INTER_CUBIC)

    return downscaled_img
