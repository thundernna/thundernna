import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
#from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset
BatchSize = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time
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


def load_imagenet(PATH = "./data/"):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = torchvision.datasets.ImageNet(root=PATH, split='val',
                                            transform=transform)
    return dataset

def predict_torch(model, img):
    with torch.no_grad():
        torch_output = model(img)
    return torch_output


def thundernna_attack(img, target, model, epsilon):
    model.eval()
    img.requires_grad = True
    output = predict_torch(model, img)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = img.grad.data
    tub = torch.clamp(torch.ones_like(img).to(device) / data_grad, -epsilon, epsilon)
    tub = torch.nan_to_num(tub)
    return img + tub

pretrained_model = models.resnet18(pretrained=True,  progress = True).to(device)
pretrained_model.eval()

dataset = load_imagenet()

test_loader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize, shuffle=False)
epsilon = 0.3
correct = 0
start = time.time()
for idx, (img, target) in tqdm(enumerate(test_loader)):
    perturbed_img =thundernna_attack(img, target, pretrained_model, epsilon)
    out = predict_torch(pretrained_model, perturbed_img)
    final_pred = out.data.max(1, keepdim=True)[1]
    correct += final_pred.eq(target.data.view_as(final_pred)).sum()
end = time.time()
print("using",end-start ,"s")
final_acc = correct/float(len(test_loader.dataset))
print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader.dataset), final_acc))


