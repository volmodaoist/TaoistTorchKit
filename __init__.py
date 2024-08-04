'''
TaoistTorchKit 包含了若干子模块，
  (1) 首先导入本模块会首先执行 __init__.py，我们在此导入了这些子模块，
  (2) 然后这些子模块会进一步查找目录下面的 __init__.py， 继续导入我们需要的函数

因而我们可以使用下面的句式调用我们想用的函数:
    ttk.submodule1.function
    ttk.submodule2.function
'''

from . import plt
from . import utils
from . import light
from . import dataset
from . import modules


# import os, random, torch
# DATASET_PATH = '/home/public-datasets'
# GPU = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# CPU = torch.device("cpu")