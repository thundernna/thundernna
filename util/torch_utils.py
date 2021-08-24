import os
import numpy as np
import torch
import torchvision


def recreate_image(im_as_var):
    """
        Recreates images from a torch variable, sort of reverse preprocessing

    Args:
        im_as_var (torch variable): Image to recreate

    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = torch.clone(im_as_var)
    recreated_im = recreated_im.cpu().numpy()[0]
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im


def predict_torch(model, img):
    # with torch.no_grad():
    torch_output = model(img)
    
    return torch_output


def predict_torch_ng(model, img):
    with torch.no_grad():
        torch_output = model(img)
    
    return torch_output


# def predict_torch(model, img):
#     with torch.no_grad():
#         torch_img = torch.from_numpy(img)
#         torch_output = model(torch_img)
    
#     return torch_output.numpy()
