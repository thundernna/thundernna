import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
sys.path.append("..")
from util.torch_utils import *
from engine import *


def thundernna_attack(img, target, model, epsilon):
    img.requires_grad = True
    output = predict_torch(model, img)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = img.grad.data
    tub = torch.clamp(torch.ones_like(img) / data_grad, -epsilon, epsilon)
    tub = torch.nan_to_num(tub)
    return img, tub


def thundernna_with_engine(rt, imgType, ctx, img, target, model, epsilon):
    model.eval()
    img.requires_grad = True
    output = predict(rt, img, imgType, ctx)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = img.grad.data
    tub = torch.clamp(torch.ones_like(img) / data_grad, -epsilon, epsilon)
    tub = torch.nan_to_num(tub)
    return img, tub

