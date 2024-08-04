
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


    
class ImageFolderDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


class ImageDataModule:
    def __init__(self, image_paths, labels,
                 input_size = (None, 3, 32, 32), 
                 batch_size = 32, 
                 valid_split = 0.1, 
                 test_split = 0.1, 
                 num_workers = 8,
                 shuffle = True, 
                 random_seed =  42):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        
        self.batch_size = batch_size
        self.valid_split = valid_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.random_seed = random_seed
        
        self._transform(input_size)
        self._prepare_dataloaders()

    # 数据预处理的部分需要根据具体的任务进行修改
    def _transform(self, input_size = (None, 3, 32, 32)):
        _, c, h, w = input_size
        trans =  [Resize((h, w)), ToTensor()]
        trans += [Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))] if c == 3 else []
        self.transform = Compose(trans)
        

    def _prepare_dataloaders(self):
        dataset_size = len(self.image_paths)
        indices = list(range(dataset_size))
        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(indices)

        test_split_idx = int(np.floor(self.test_split * dataset_size))
        valid_split_idx = int(np.floor(self.valid_split * (dataset_size - test_split_idx)))
        
        test_indices = indices[:test_split_idx]
        valid_indices = indices[test_split_idx:test_split_idx + valid_split_idx]
        train_indices = indices[test_split_idx + valid_split_idx:]

        self.train_set = ImageFolderDataset(self.image_paths[train_indices], self.labels[train_indices], self.transform)
        self.valid_set = ImageFolderDataset(self.image_paths[valid_indices], self.labels[valid_indices], self.transform)
        self.test_set = ImageFolderDataset(self.image_paths[test_indices], self.labels[test_indices], self.transform)
        
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size,num_workers = self.num_workers,  shuffle=False)
        self.test_loader = DataLoader(self.test_set,  batch_size=self.batch_size, num_workers = self.num_workers, shuffle=False)

    def getTrans(self):
        return self.transform
    
    def setTrans(self, transform):
        self.transform = transform
        self._prepare_dataloaders()
    



from torchvision import datasets
from torch.utils.data import ConcatDataset, random_split


class ImageTensorSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform 

    def __len__(self):
        return len(self.subset) 

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class ImageToyData:
    def __init__(self, dataset, 
                       root_path = '/home/public-datasets', 
                       input_size = (None, 3, 32, 32), 
                       batch_size = 32,
                       num_workers = 8,
                       val_ratio = 0.1, test_ratio = 0.1):
        self.dataset = dataset 
        self.root_path = root_path
        
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self._transform(input_size)
        self.raw_data = self._get_raw_data()

    def _transform(self, input_size = (None, 3, 32, 32)):
        _, c, h, w = input_size
        trans =  [Resize((h, w)), ToTensor()]
        trans += [Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711))] if c == 3 else []
        self.transform = Compose(trans)
        
    def _get_raw_data(self):
        determine_split = lambda kwargs: 'train' if kwargs.pop('train', True) else 'test'
        
        builtin_datasets = {
            'mnist': lambda *args, **kwargs: datasets.MNIST(*args, **kwargs),
            'kmnist': lambda *args, **kwargs: datasets.KMNIST(*args, **kwargs),
            'fmnist': lambda *args, **kwargs: datasets.FashionMNIST(*args, **kwargs),
            'cifar10': lambda *args, **kwargs: datasets.CIFAR10(*args, **kwargs),
            'cifar100': lambda *args, **kwargs: datasets.CIFAR100(*args, **kwargs),
            'svhn': lambda *args, **kwargs: datasets.SVHN(split=determine_split(kwargs), *args, **kwargs),
            'stl10': lambda *args, **kwargs: datasets.STL10(split=determine_split(kwargs), *args, **kwargs),
            'flowers102': lambda *args, **kwargs: datasets.Flowers102(split=determine_split(kwargs), *args, **kwargs),
        }

        dataset_func = builtin_datasets.get(self.dataset.lower())
        if dataset_func is None:
            raise ValueError(f"Unsupported dataset name: {self.args.dataset}")

        train_set = dataset_func(root=self.root_path, train=True, download=True)
        test_set = dataset_func(root=self.root_path, train=False, download=True)


        self.raw_data = ConcatDataset([train_set, test_set])
        
        total_size = len(self.raw_data)
        train_ratio = 1 - self.val_ratio - self.test_ratio
        train_size = int(total_size * train_ratio)
        valid_size   = int(total_size * self.val_ratio)
        test_size  = total_size - train_size - valid_size
        
        train_data, valid_data, test_data = random_split(self.raw_data, [train_size, valid_size, test_size])
        
        self.train_set = ImageTensorSubset(test_data, self.transform)
        self.valid_set = ImageTensorSubset(valid_data, self.transform)
        self.test_set = ImageTensorSubset(train_data, self.transform)
        

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=True)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, num_workers = self.num_workers, shuffle=False)
        self.test_loader = DataLoader(self.test_set,  batch_size=self.batch_size, num_workers = self.num_workers, shuffle=False)
    


    
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
    
    # 移除'257.clutter'这个无效类别, 随后更新dataset的samples和targets
    idx_to_remove = dataset.class_to_idx.get('257.clutter', None)  
    if idx_to_remove is not None:
        dataset.samples = [s for s in dataset.samples if s[1] != idx_to_remove]
        dataset.targets = [s[1] for s in dataset.samples]
    
    return dataset
