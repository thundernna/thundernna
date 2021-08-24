import os
import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time

from numpy.core.numeric import require
import torch.nn as nn
import torch.nn.functional as F

from util.torch_utils import *

DUP = 1
INP_NUM = 1000


def preprocess(img):
    my_preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = my_preprocess(img)
    img = np.expand_dims(img, 0)
    return img


def image_repo(num):
    imagePath = "./images/animals10/raw-img/gatto/"
    imagePath = os.path.join(os.path.abspath(os.curdir), imagePath)
    files = os.listdir(imagePath)

    repo = []
    for i, imgName in enumerate(files[:max(num, 0)], 1):
        print("Preparing #%4d input picture..." % i)
        img = Image.open(os.path.join(imagePath, imgName)).resize((224, 224))
        repo.append(torch.from_numpy(preprocess(img)))
    
    print("Input images ready. ")
    return repo


def thundernna_attack(img, model, epsilon):
    img.requires_grad = True
    output = predict_torch(model, img)
    # target = torch.tensor([torch.argmax(output)])
    target = torch.argmax(output).reshape(1)
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


def pure():
    model_name = args.model
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    if args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda:0"
    torch.device(device)

    if args.cmp:
        repo = image_repo(INP_NUM)
        print("%d input pictures prepared" % len(repo))

        print("Profiling...")
        duration_pure, duration_engine = 0, 0
        for i in range(DUP):
            print("Dup #%4d times..." % (i + 1))
            start = time()
            for img in repo:
                img, turb = thundernna_attack(img, model, args.epsilon)
            end = time()
            duration_pure += end - start
        
        print("\nConfig:\n    model = %s\n    INP_NUM = %d\n    DUP = %d" % (model_name, INP_NUM, DUP))
        print("Profiling Result")
        print("================================================")
        print("    Pure Algorithm:       ", duration_pure, "s")
        print("    With Inference Engine:", duration_engine, "s")
        print("================================================\n")

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--attacker", type=str, default="thunder")
    parser.add_argument("--compiled", type=int, default=0)
    parser.add_argument("--cc", type=int, default=1)
    parser.add_argument("--cmp", type=int, default=0)
    parser.add_argument("--prof_cuda", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.2)

    args = parser.parse_args()

    pure()