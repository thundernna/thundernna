import os
from time import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
import tvm.contrib.graph_runtime as graph_runtime

import sys
sys.path.append("..")
from util import cudaprofile as cp
from util.img_utils import *
from util.torch_utils import *


BUILD_PATH = "build/"
INP_NUM = 1000
DUP = 1

BUILD_PATH = os.path.join(os.getcwd(), BUILD_PATH)
if not os.path.exists(BUILD_PATH):
    os.makedirs(BUILD_PATH)
print("BUILD_PATH SET:", BUILD_PATH, "\n")


def main(model_name, model, rt):
    def warm_up():
        lth = len(repo)
        tvm_output_list, torch_output_list = [], []
        print("Warming up...")
        for i, img in enumerate(repo):
            print("Warming up... %4d / %4d" % ((i + 1), lth))
            tvm_output_list.append(predict_torch(model, img))
            torch_output_list.append(predict(rt, img))
        
        consistFlag = synset_lookup(tvm_output_list, torch_output_list)
        result = "True" if consistFlag else "False"
        print("\nResult of consistency check:", result)
        return consistFlag
    

    def e2e():
        # cp.start()

        start = time()
        # for img in repo:
        #     predict_torch(model, img)
        end = time()
        tem_torch = end - start

        # cp.stop()

        cp.start()

        start = time()
        for img in repo:
            predict(rt, img)
        end = time()
        tem_tvm = end - start

        cp.stop()

        return tem_torch, tem_tvm


    repo = image_repo(INP_NUM)
    print("%d input pictures prepared" % len(repo))

    # consistFlag = warm_up()

    print("End to End profiling...")
    duration_torch, duration_tvm = 0, 0
    for i in range(DUP):
        print("Dup #%4d times..." % (i + 1))
        tem_torch, tem_tvm = e2e()
        duration_torch += tem_torch
        duration_tvm += tem_tvm

    print("\nConfig:\n    model = %s\n    INP_NUM = %d\n    DUP = %d" % (model_name, INP_NUM, DUP))
    print("End2End Profiling Result")
    print("================================================")
    print("    PyTorch RunTime:  ", duration_torch, "s")
    print("    TVM Graph RunTime:", duration_tvm, "s")
    print("================================================\n")

    return


def init(cplFlag=False, device="cpu"):
    print("Eninge initializing...")

    img = sample_img()
    if device == "cpu":
        target = "cpu"
        ctx = tvm.cpu()
    else:
        ctx = tvm.gpu()
        target = "cuda -libs=cublas,cudnn"
    
    rt = runtime_v7(ctx=ctx, target=target, compiled=cplFlag)
    tvm_output = predict(rt, img, imgType="default", ctx=ctx)

    torch_output = predict_torch(model, img)
    consistFlag = synset_lookup([tvm_output], [torch_output])


    print("Init Done.")
    result = "True" if consistFlag else "False"
    print("Result of consistency check:", result, "\n")
    return rt


def load_relay_module(graph_def, input_shape): 
    print("GraphDef -> Relay")
    print(type(graph_def))

    # Frontend: GraphDef -> Relay
    mod, params = relay.frontend.from_pytorch(graph_def, input_infos=input_shape)
    mod = relay.transform.InferType()(mod)
    return mod, params


def encapsulate(model, img, imgType="default"):
    # We grab the TorchScripted model via tracing
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    shape_list = [(imgType, img.shape)]
    mod, params = load_relay_module(scripted_model, shape_list)
    return mod, params


def compile_graph_runtime(model_name, mod, params, target="llvm", target_host="llvm"):
    if target != "llvm":
        target = "cuda -libs=cublas,cudnn"
    else:
        target = tvm.target.Target("llvm")
        dev = tvm.cpu(0)

    graph, lib, params = relay.build(mod, target_host=target_host, target=target, params=params)
    os.system("")
    lib_file = "%s.lib.tar" % os.path.join(BUILD_PATH, model_name)
    graph_file = "%s.graph.json" % os.path.join(BUILD_PATH, model_name)
    params_file = "%s.params.bin" % os.path.join(BUILD_PATH, model_name)

    lib.export_library(lib_file)
    with open(graph_file, 'w') as f:
        f.write(graph)
    with open(params_file, 'wb') as f:
        f.write(relay.save_param_dict(params))
    print("save done")
    return graph, lib, params



def load_graph_runtime(model_name):
    lib_file = "%s.lib.tar" % os.path.join(BUILD_PATH, model_name)
    graph_file = "%s.graph.json" % os.path.join(BUILD_PATH, model_name)
    params_file = "%s.params.bin" % os.path.join(BUILD_PATH, model_name)

    lib = tvm.runtime.module.load_module(lib_file)
    with open(graph_file, 'r') as f:
        graph = f.read()
    with open(params_file, 'rb') as f:
        params = bytearray(f.read())

    return graph, lib, params


def runtime_v7(model_name, model, ctx=tvm.cpu(), target="llvm", compiled=False):
    if not compiled:
        img = sample_img()
        mod, params = encapsulate(model, img)
        graph, lib, params = compile_graph_runtime(model_name, mod, params, target=target, target_host="llvm")

    graph, lib, params = load_graph_runtime(model_name)
    rt = graph_runtime.create(graph, lib, ctx)
    rt.load_params(params)
    print("load done")
    return rt
    

def predict(rt, img, imgType="default", ctx=tvm.cpu()):
    rt.set_input(imgType, tvm.nd.array(img.astype("float32")))
    rt.run()
    ctx.sync()
    tvm_output = rt.get_output(0)
    return tvm_output.asnumpy()

