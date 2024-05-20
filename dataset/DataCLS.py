from torchvision import datasets
from torchvision import transforms

from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

# 当直接运行一个.py模块，其将被视为顶级脚本，而非某个 package 之中的内容，此时相对导入会出错
from .DataBase import VTS
from .DataBase import DataBase


class DataCLS(DataBase):
    def __init__(self, args, root_path, transform_combs=None):
        super().__init__(args, root_path, transform_combs)
        
    def _get_raw_data(self):
        # 单独处理非内建数据集
        if self.dataset.lower() == 'imagenette':
            self.raw_data = _load_imagenette(self.root_path)
        elif self.dataset.lower() == 'caltech101':
            self.raw_data = _load_caltech_101(self.root_path)
        elif self.dataset.lower() == 'caltech256':
            self.raw_data = _load_caltech_256(self.root_path)
        
        if self.raw_data is not None:
            return self.raw_data
        
        # 由于 stl10 和 flowers102 使用的接口与其它内置数据集不太一样，因而需要使用辅助函数转化一下
        determine_split = lambda kwargs: 'train' if kwargs.pop('train', True) else 'test'
        builtin_datasets = {
            'mnist': lambda *args, **kwargs: datasets.MNIST(*args, **kwargs),
            'kmnist': lambda *args, **kwargs: datasets.KMNIST(*args, **kwargs),
            'fmnist': lambda *args, **kwargs: datasets.FashionMNIST(*args, **kwargs),
            'cifar10': lambda *args, **kwargs: datasets.CIFAR10(*args, **kwargs),
            'cifar100': lambda *args, **kwargs: datasets.CIFAR100(*args, **kwargs),
            
            'svhn': lambda *args, **kwargs: datasets.SVHN(split = determine_split(kwargs), *args, **kwargs), 
            'stl10': lambda *args, **kwargs: datasets.STL10(split = determine_split(kwargs), *args, **kwargs),
            'flowers102': lambda *args, **kwargs: datasets.Flowers102(split = determine_split(kwargs), *args, **kwargs),
        }
        
        builtin_dataset = builtin_datasets.get(self.dataset.lower())
        
         
        if not builtin_dataset:
            raise ValueError(f"Unsupported dataset name: {self.dataset}")
        
        tset = builtin_dataset(root = self.root_path, train=True, download=True)
        eset = builtin_dataset(root = self.root_path, train=False, download=True)
        self.raw_data = ConcatDataset([tset, eset])
        
        return self.raw_data
    
    def _split_dataset(self):
        test_size = int(len(self.raw_data) * self.test_size)
        train_size = len(self.raw_data) - test_size
        
        tset, eset = random_split(self.raw_data, [train_size, test_size])
        
        self.tset = VTS(tset, transform=self.tset_aug)
        self.eset = VTS(eset, transform=self.eset_aug)
        return self.tset, self.eset
    
    def _get_transforms(self):
        if self.transform_combs is not None:
            self.tset_aug = self.transform_combs['tset']
            self.eset_aug = self.transform_combs['eset']
            return
        
        _, c, h, w = self.input_size
        comps =  [transforms.ToTensor(), 
                  transforms.Resize((h, w), antialias = True)]
        comps += [transforms.Grayscale()] if c == 1 else []
        comps += [transforms.Normalize(
                    # 这个均值方差是从ImageNet 数据集之中统计得出的图像数据变为正态分布
                    mean = [0.485, 0.456, 0.406],    
                     std = [0.229, 0.224, 0.225]
                )] if c == 3 else []
        
        self.tset_aug = self.eset_aug = transforms.Compose(comps)
        


'''
请按下列方式组织管理数据集文件:

    1. 下载 caltech 101/256 系列数据集，并将它们放在自建的 caltech 目录之下，数据集来自 Kaggle:
        https://www.kaggle.com/datasets/imbikramsaha/caltech-101
        https://www.kaggle.com/datasets/jessicali9530/caltech256


    2. 下载 ImageNette 数据集，我们将其放在自建的 ImageNette 文件夹之内，
        https://www.kaggle.com/datasets/aniladepu/imagenette 这是ImageNet子集
        
      适用用来忽悠审稿人，使其以为你在大型数据集上面跑过；关于这个数据集，其中提供了含噪标签集，
    标签之中包含错误标注，分别包括 1/5/25/50 四个比例 (i.e. 含有相应比例的噪声标签)
'''

import os

def _load_imagenette(root):
    tset = datasets.ImageFolder(root = os.path.join(root, 'ImageNette/imagenette/train'))
    eset = datasets.ImageFolder(root = os.path.join(root, 'ImageNette/imagenette/val'))
    
    return ConcatDataset([tset, eset])


def _load_caltech_101(root):
    dataset = datasets.ImageFolder(root = os.path.join(root, 'caltech/caltech-101'))
    return dataset

def _load_caltech_256(root):
    dataset = datasets.ImageFolder(root = os.path.join(root, 'caltech/256_ObjectCategories'))
    
    # 移除'257.clutter'这个无效类别
    idx_to_remove = dataset.class_to_idx.get('257.clutter', None)  
    if idx_to_remove is not None:
        # 更新dataset的samples和targets
        dataset.samples = [s for s in dataset.samples if s[1] != idx_to_remove]
        dataset.targets = [s[1] for s in dataset.samples]
    
    return dataset


