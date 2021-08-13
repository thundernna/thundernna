import os
import ssl
from time import time
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
# from tvm.contrib import graph_executor
import tvm.contrib.graph_runtime as graph_runtime
import cudaprofile as cp


BUILD_PATH = "build/"
INP_NUM = 1000
DUP = 2


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
        start = time()
        for img in repo:
            predict_torch(model, img)
        end = time()
        tem_torch = end - start

        start = time()
        for img in repo:
            predict(rt, img)
        end = time()
        tem_tvm = end - start

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

    print("\nConfig:\n  model = %s\n  INP_NUM = %d\n  DUP = %d" % (model_name, INP_NUM, DUP))
    print("End2End Profiling Result")
    print("================================================")
    print("PyTorch RunTime:  ", duration_torch, "s")
    print("TVM Graph RunTime:", duration_tvm, "s")
    print("================================================\n")

    return


def init():
    img = sample_img()
    # mod, params = encapsulate(model, img)
    # tvm_output = runtime(mod, params, img, imgName="default" target="llvm")

    # ctx = tvm.cpu()
    # target = "llvm"
    ctx = tvm.gpu()
    target = "cuda -libs=cublas,cudnn"
    
    # rt = runtime_v7(ctx=tvm.cpu(), target="llvm", compiled=False)
    rt = runtime_v7(ctx=ctx, target=target, compiled=False)
    # rt = runtime_v7(ctx=ctx, target=target, compiled=True)
    tvm_output = predict(rt, img, imgType="default", ctx=ctx)

    torch_output = predict_torch(model, img)
    synset_lookup([tvm_output], [torch_output])

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


def compile_graph_runtime(mod, params, target="llvm", target_host="llvm"):
    if target != "llvm":
        target = "cuda -libs=cublas,cudnn"
    else:
        target = tvm.target.Target("llvm")
        dev = tvm.cpu(0)

    graph, lib, params = relay.build(mod, target_host=target_host, target=target, params=params)
    lib_file = "%s.lib.tar" % os.path.join(BUILD_PATH, "ResNet-18")
    graph_file = "%s.graph.json" % os.path.join(BUILD_PATH, "ResNet-18")
    params_file = "%s.params.bin" % os.path.join(BUILD_PATH, "ResNet-18")

    lib.export_library(lib_file)
    with open(graph_file, 'w') as f:
        f.write(graph)
    with open(params_file, 'wb') as f:
        f.write(relay.save_param_dict(params))
    print("save done")
    return graph, lib, params


# def runtime(mod, params, img, imgName="default", target="llvm", ):
#     from tvm.contrib import graph_executor
#     if target == "gpu":     # Preserved
#         target = tvm.target.Target("gpu", host="gpu")
#     else:
#         target = tvm.target.Target("llvm", host="llvm")
#         dev = tvm.cpu(0)
    
#     with tvm.transform.PassContext(opt_level=3):
#         lib = relay.build(mod, target=target, params=params)
    
#     dtype = "float32"
#     m = graph_executor.GraphModule(lib["default"](dev))
#     # Set inputs
#     m.set_input(imgName, tvm.nd.array(img.astype(dtype)))
#     # Execute
#     m.run()
#     # Get outputs
#     tvm_output = m.get_output(0)
#     return tvm_output


def load_graph_runtime():
    lib_file = "%s.lib.tar" % os.path.join(BUILD_PATH, "ResNet-18")
    graph_file = "%s.graph.json" % os.path.join(BUILD_PATH, "ResNet-18")
    params_file = "%s.params.bin" % os.path.join(BUILD_PATH, "ResNet-18")

    lib = tvm.runtime.module.load_module(lib_file)
    with open(graph_file, 'r') as f:
        graph = f.read()
    with open(params_file, 'rb') as f:
        params = bytearray(f.read())

    return graph, lib, params


def runtime_v7(ctx=tvm.cpu(), target="llvm", compiled=False):
    if not compiled:
        img = sample_img()
        mod, params = encapsulate(model, img)
        graph, lib, params = compile_graph_runtime(mod, params, target=target, target_host="llvm")

    graph, lib, params = load_graph_runtime()
    rt = graph_runtime.create(graph, lib, ctx)
    rt.load_params(params)
    print("load done")
    return rt
    

def predict(rt, img, imgType="default", ctx=tvm.cpu()):
    dtype = "float32"

    rt.set_input(imgType, tvm.nd.array(img.astype(dtype)))
    rt.run()
    ctx.sync()
    tvm_output = rt.get_output(0)
    return tvm_output.asnumpy()


def predict_torch(model, img):
    with torch.no_grad():
        torch_img = torch.from_numpy(img)
        torch_output = model(torch_img)
    
    return torch_output.numpy()


def synset_lookup(tvm_output_list, torch_output_list):
    ssl._create_default_https_context = ssl._create_unverified_context
    synset_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_synsets.txt",
        ]
    )
    synset_name = "imagenet_synsets.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synsets = f.readlines()

    synsets = [x.strip() for x in synsets]
    splits = [line.split(" ") for line in synsets]
    key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

    class_url = "".join(
        [
            "https://raw.githubusercontent.com/Cadene/",
            "pretrained-models.pytorch/master/data/",
            "imagenet_classes.txt",
        ]
    )
    class_name = "imagenet_classes.txt"
    class_path = download_testdata(class_url, class_name, module="data")
    with open(class_path) as f:
        class_id_to_key = f.readlines()

    class_id_to_key = [x.strip() for x in class_id_to_key]

    consistFlag = True
    for i in range(min(len(tvm_output_list), len(torch_output_list))):
        # Get top-1 result for TVM
        top1_tvm = np.argmax(tvm_output_list[i])
        tvm_class_key = class_id_to_key[top1_tvm]

        # Get top-1 result for PyTorch
        top1_torch = np.argmax(torch_output_list[i])
        torch_class_key = class_id_to_key[top1_torch]

        print("\nImage #%4d" % (i + 1))
        print("Relay Prediction id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
        print("Torch Prediction id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
        if tvm_class_key != torch_class_key:
            consistFlag = False
    
    return consistFlag


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


def sample_img():
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")
    img = Image.open(img_path).resize((224, 224))
    return preprocess(img)


def image_repo(num):
    imagePath = "/workdir/shengze/ResNet/thundernna/images/animals10/raw-img/gatto/"
    files = os.listdir(imagePath)

    repo = []
    for i, imgName in enumerate(files[:max(num, 0)], 1):
        print("Preparing #%4d input picture..." % i)
        img = Image.open(os.path.join(imagePath, imgName)).resize((224, 224))
        repo.append(preprocess(img))
    
    print("Input images ready. ")
    return repo



if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    print(os.getcwd())
    BUILD_PATH = os.path.join(os.getcwd(), BUILD_PATH)
    os.system("mkdir -p %s" % BUILD_PATH)
    print(BUILD_PATH)

    # model_name = "resnet18"
    # model_name = "resnet34"
    # model_name = "resnet50"
    model_name = "resnet101"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    model = model.eval()

    rt = init()
    main(model_name, model, rt)
