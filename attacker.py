import numpy as np
from numpy.core.numeric import require
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import sys
sys.path.append("..")
from util.torch_utils import *
from engine import *


def thundernna_attack(img, model, epsilon):
    img.requires_grad = True
    output = predict_torch(model, img)
    # target = torch.tensor([torch.argmax(output)])
    target = torch.argmax(output).reshape(1)
    print(target)
    # output = Variable(output, requires_grad=True)

    loss = F.nll_loss(output, target)
    # loss.requires_grad = True
    model.zero_grad()
    loss.backward()
    data_grad = img.grad.data
    tub = torch.div(torch.ones_like(img), data_grad, out=torch.zeros_like(img))
    tub = torch.clamp(tub, -epsilon, epsilon)
    # tub = torch.clamp(torch.ones_like(img) / data_grad, -epsilon, epsilon)
    # tub = torch.nan_to_num(tub)
    
    return img, tub


def thundernna_with_engine(rt, imgType, ctx, img, model, epsilon):
    img.requires_grad = True
    output = torch.from_numpy(predict(rt, img.detach().numpy(), imgType, ctx))
    target = torch.argmax(torch.Tensor(output)).reshape(1)
    loss = F.nll_loss(output, target)
    # loss.requires_grad = True
    model.zero_grad()
    loss.backward()
    data_grad = img.grad.data
    tub = torch.clamp(torch.ones_like(img) / data_grad, -epsilon, epsilon)
    tub = torch.nan_to_num(tub)
    return img, tub

