import os
import argparse
import numpy as np
import torch
import torchvision
from PIL import Image
import tvm
from time import time

from util import cudaprofile as cp
from attacker import *
from engine import *

OUT_PATH = "/output"
OUT_PATH = os.path.join(os.getcwd(), OUT_PATH)
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)
print("OUT_PATH SET:", OUT_PATH, "\n")


def entrance():
    def compare():
        start = time()
        for img in repo:
            img, turb = thundernna_attack(img, target, model, args.epsilon)
        end = time()
        tem_torch = end - start

        if args.prof_cuda:
            cp.start()

        start = time()
        for img in repo:
            img, turb = thundernna_with_engine(rt, "default", ctx, img, target, model, args.epsilon)
        end = time()
        tem_tvm = end - start

        if args.prof_cuda:
            cp.stop()

        return tem_torch, tem_tvm


    model_name = args.model
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    if args.device == "cpu":
        device = "cpu"
        ctx = tvm.cpu()
    else:
        device = "cuda:0"
        ctx = tvm.gpu()
    torch.device(device)

    cplFlag = True if args.compiled else False
    rt = init(cplFlag, device)

    ### main(model_name, model, rt)


    if args.cc:
        img = sample_img()

        img1, turb1 = thundernna_attack(img, target, model, args.epsilon)
        img2, turb2 = thundernna_with_engine(rt, "default", ctx, img, target, model, args.epsilon)

        mob1 = predict_torch(model, img1 + turb1)
        mob2 = predict_torch(model, img2 + turb2)
        consistFlag = np.argmax(mob1) == np.argmax(mob2)

        result = "True" if consistFlag else "False"
        print("Consistency of disturbed images:", result, "\n")

        im1 = Image.fromarray(img1 + turb1)
        im2 = Image.fromarray(img2 + turb2)
        im1.save(os.path.join(OUT_PATH, "ori_disturbed.jpeg"))
        im2.save(os.path.join(OUT_PATH, "engine_disturbed.jpeg"))
        print("Disturbed images saved.")
    
    if args.cmp:
        repo = image_repo(INP_NUM)
        print("%d input pictures prepared" % len(repo))

        print("Profiling...")
        duration_pure, duration_engine = 0, 0
        for i in range(DUP):
            print("Dup #%4d times..." % (i + 1))
            tem_torch, tem_tvm = compare()
            duration_pure += tem_torch
            duration_engine += tem_tvm
        
        print("\nConfig:\n    model = %s\n    INP_NUM = %d\n    DUP = %d" % (model_name, INP_NUM, DUP))
        print("Profiling Result")
        print("================================================")
        print("    Pure Algorithm:       ", duration_pure, "s")
        print("    With Inference Engine:", duration_engine, "s")
        print("================================================\n")



    return 


def dumb():
    repo = image_repo(INP_NUM)
    print("%d input pictures prepared" % len(repo))

    print("End to End profiling...")
    duration_torch, duration_tvm = 0, 0
    for i in range(DUP):
        print("Dup #%4d times..." % (i + 1))
        tem_torch, tem_tvm = e2e()
        duration_torch += tem_torch
        duration_tvm += tem_tvm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--build_dir", type=str, default='.')
    # parser.add_argument("--consistency_check", type=int, default=0)
    # parser.add_argument("--benchmark", type=int, default=1)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--attacker", type=str, default="thunder")
    parser.add_argument("--compiled", type=int, default=0)
    parser.add_argument("--cc", type=int, default=1)
    parser.add_argument("--cmp", type=int, default=1)
    parser.add_argument("--prof_cuda", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.2)

    args = parser.parse_args()

    entrance()
