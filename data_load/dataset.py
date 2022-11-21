# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/18 17:59
 @name: 
 @desc:
"""

import torchvision.datasets as datasets
import torchvision.transforms as transforms

# data_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_set = datasets.ImageFolder(root='E:/Datasets/imagenet/train', transform=transform)
val_set = datasets.ImageFolder(root='E:/Datasets/imagenet/val', transform=transform)
test_set = datasets.ImageFolder(root='E:/Datasets/imagenet/test', transform=transform)
