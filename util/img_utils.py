import os
import ssl
import numpy as np
from torchvision import transforms
from PIL import Image
from tvm.contrib.download import download_testdata


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
    imagePath = "../images/animals10/raw-img/gatto/"
    imagePath = os.path.join(os.path.abspath(os.curdir), imagePath)
    files = os.listdir(imagePath)

    repo = []
    for i, imgName in enumerate(files[:max(num, 0)], 1):
        print("Preparing #%4d input picture..." % i)
        img = Image.open(os.path.join(imagePath, imgName)).resize((224, 224))
        repo.append(preprocess(img))
    
    print("Input images ready. ")
    return repo

