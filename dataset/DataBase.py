from typing import Tuple
from argparse import Namespace
from torchvision import transforms

from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# 创建子数据集类 (VTS, VisionTransformedSubset)，并把 transform 用到每个数据项
class VTS(Dataset):
    def __init__(self, subset: Subset, transform: transforms.Compose = None):
        self.subset = subset
        self.transform = transform 
            
    def __len__(self):
        return len(self.subset)  
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        return self.transform(x), y
    
    
class DataBase:
    def __init__(self, args, root_path, transform_combs: dict = None):
        if isinstance(args, dict):
            args = Namespace(**args)
             
        # 通过命令行获取的参数
        self.seed = args.seed
        self.input_size = args.input_size
        self.batch_size = args.batch_size
        self.test_size = args.test_size
        self.num_workers = args.num_workers
        self.dataset = args.dataset
        
        # 设置数据集路径放置的位置
        self.root_path = root_path
        self.transform_combs = transform_combs
        
        self.raw_data = None 
        self.tset, self.tset_aug = None, None
        self.eset, self.eset_aug = None, None
    
        # 下面执行顺序不可以更改
        self._get_raw_data()
        self._get_transforms()
        self._split_dataset()
   
    # 获取未经加工的数据集
    def _get_raw_data(self) -> Dataset:
        raise NotImplementedError
        
    # 使用数据转化器来做数据增强
    def _get_transforms(self):
        raise NotImplementedError
    
    # 使用数据预处理器加工的数据集，随后进行划分
    def _split_dataset(self) -> Tuple[Subset, Subset]:
        raise NotImplementedError
   
    
    def get_loader(self, is_train = True):
        dataset = self.tset if is_train else self.eset
        return DataLoader(dataset, batch_size = self.batch_size, 
                                   num_workers = self.num_workers, 
                                   shuffle = is_train)
