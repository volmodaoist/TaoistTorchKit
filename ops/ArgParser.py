import torch
import argparse
import numpy as np
import os
import random
import logging


GPU = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CPU = torch.device("cpu")

# 任务超参数设置模块
class ArgParser(argparse.ArgumentParser):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__dict__['_initialized'] = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
                    
        ''' 重要注释:
        
            我们希望能够直接通过成员运算访问 ArgParser.args 里面的属性，因而我们需要重写 __setter__ 转发属性访问。
            
            然而一旦重写了这魔术方法，那些直接使用 . 成员运算符赋值也会调用魔术方法 setter, 所以我们构造函数之中初始
            设置也会调用我们的重写的 setter，而它访问的 self.args 此时未被创建，找不到属性继续转发，导致无限递归。
            为此我们使用 self.__dict__直接设置几个关键属性，避免出现以上情况。
            如无必要，请勿改写!
        '''
        
        self.__dict__['parser'] = argparse.ArgumentParser()
        
        # 通用性参数
        self.parser.add_argument("--model", default = None, type = str,              # 模型
                                help = "The model needed for the specific task.")
        self.parser.add_argument("--backbone", default = None, type = str,           # 基座模型
                                help = "The backbone for the specific model.")
        self.parser.add_argument("--dataset", default = None, type = str,            # 数据集
                                help = "Dataset for train/test model.")
        self.parser.add_argument("--seed", default = 3407, type = int,               # 随机数设置
                                 help = "Random seed for initializing training.")
        self.parser.add_argument("--mode", default = 10, type = int,                 # 设置训练模型和测试模块
                                 help = "Setting for test (0), train (1), or both (10) model, ")
        self.parser.add_argument("--device", default = GPU, type = torch.device,     # 模式:使用GPU或CPU
                                 help = "Select GPU/CPU mode.")
        self.parser.add_argument("--train-remark", default = 'NT', type = str,       # 训练模型阶段的注释
                                 help = "Comment for training mode.")
        self.parser.add_argument("--eval-remark", default = 'NT', type = str,        # 测试模型阶段的注释
                                 help = "Comment for evaling mode.")      

        '''
        基本参数设置:
         - 主要是模型训练阶段的优化器、调度器参数,
         - 其次是数据集读入参数，包括读取线程、数据尺寸,
        '''
        
        # 设置学习率
        self.parser.add_argument("-lr", "--learning-rate", default = 0.01, type = float,
                                 help = "Epoch for train model.")
        
        # 设置学习率调度器的衰减参数，这个参数用来决定每隔多少步衰减学习率
        self.parser.add_argument("--lr-decay-step", default = 10, type = int,        
                            help = "Used to decide how many steps to decay the learning rate every.")
        
        # 设置学习率调度器的衰减参数，这个参数决定每次学习率衰减的幅度是多少
        self.parser.add_argument("--lr-decay-gamma", default = 0.5, type = float,
                            help = "Used to decide how much the learning rate decays each time.")
        
        # 优化器的权重衰减
        self.parser.add_argument("--weight-decay",  default = 1e-5, type = float, 
                                 help = "Used to decide weight decay in optimizer.")
        
        # 设置批度大小
        self.parser.add_argument("--batch-size", default = 512, type = int,
                                 help = "Batch size for train/test model.")
        
        # 设置 代纪/纪元 个数
        self.parser.add_argument("--epoch", default = 10, type = int,
                                 help = "Epoch for train model.")
        
        # 设置激活函数
        self.parser.add_argument("-A", "--activate", default = 'relu', type = str,
                                 help = "Activata-function for model.")
        
        # 设置模型按批处理读入图像的维度，由于无法直接接收元组类型，可以将其设为接收四个 int 参数
        self.parser.add_argument("--input-size", nargs = 4, type = int,              
                            help = "Set channel for images that the model reads in batches.")
        
        # 设置测试集的比例
        self.parser.add_argument("--test-size", default=0.2, type=float,
                                 help="Test set proportion.")
        
        # 数据集加载使用的线程数
        self.parser.add_argument("--num-workers", default = 8, type = int,
                            help = "Number of threads to loading dataset.")  
        
        # 模型预训练权重载入路径
        self.parser.add_argument("--load-path", default = None, type = str,
                    help = "Load the pre-trained model for evaluation.")
        
        # 数据增强的方案掩码
        self.parser.add_argument("-aug", "--aug-mask", default="000000", type=str,
                                 help = "Mask used to specify data augmentation.")
        
      
        '''
        接下来的命令行参数是针对特定任务设计的，
        每当新增任务的时候，可能需要新增或是修改一些命令!
        '''
            
        # 基座模型选取: 选择使用哪个 CNN
        self.parser.add_argument("--cnn", default = None, type = str, 
                                 help = "Use a certain CNN.")
        
        # 基座模型选取: 选择使用哪个 GNN
        self.parser.add_argument("--gnn", default = None, type = str, 
                                 help = "Use a certain GNN.")
        
        # 图神经网络相关: 图卷积模块 GraphSAGE 选择何种聚合函数 
        self.parser.add_argument("-ag", "--aggr", default = None, type = str, 
                                 help = "aggregator type for GraphNN.")
        
        # 可解释性任务: 模型决策过程转化变成 graph 过程之中保留每一层 topK 顶点
        self.parser.add_argument("--topk", default = 0.5, type = float,
                                 help = "The vertex retention rate of the graph extracted from CNN.")
         
        # 对抗攻防任务: 对抗训练使用的攻击类型
        self.parser.add_argument("-at", "--attack-type", default=None, type=str,
                                 help="Attack for adversarial train/test.")
        
        # 对抗攻防任务: 对抗训练使用的攻击约束
        self.parser.add_argument("-an", "--attack-norm", default=None, type=str,
                                 help="Attack for adversarial train/test.")
        
        # 对抗攻防任务: ALP训练阶段使用的pair 函数
        self.parser.add_argument("-pl", "--pairloss", default = "L2", type = str,
                                 help="Pairloss for adversarial logit pairing.")
         
        # 对抗攻防任务：扰动幅度大小
        self.parser.add_argument("--eps", default = "8/255", type = str,
                                 help = "Epsilon for adversarail traing.")
        
        # 分类任务: 设置分类模型需要分类的个数
        self.parser.add_argument("--num-classes", default = 10, type = int,
                    help = "The number of classes the model needs to classify.") 
        
        # 模型结构修改: 设置需要修改的 layer 编号
        self.parser.add_argument("--layer-idx", default = 1, type = int,
                    help = "The index of layer that needs to be modified.")
        
        
        # 重构任务: 编码解码结构的中间维度
        self.parser.add_argument('-dim', "--dim", default = 10, type = int,
                    help = "The dimension of latent space.")
        

        self.__dict__['args'] = self.parser.parse_args()
        self.__dict__['_initialized'] = True
        

        seed_everything(self.args.seed)
    

    def parse_args(self):
        return self.args
    
    def update_args(self, config:dict = None):
        '''此处使用 vars 获取命令行参数命名空间的字典引言，更新这个字典，
           命令参数由于引用性质也会随之更新, 如此 ArgParser 对象的可以提高通用性!
        '''
        if config is not None:
            vars(self.args).update(config)
        
        seed_everything(config.get('seed', self.args.seed))
        return self.args
    
    # 设置日志文件夹
    def setup_resfiles(self):
        res_base_dir = './res'
        subdirs = ['csv', 'figures', 'log', 'weight']
        
        paths = {}
        for subdir in subdirs:
            dir_path = os.path.join(res_base_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
            paths[subdir] = dir_path
        
        return paths

    def setup_logger(self, config: dict = None, verbose:bool = True):
        args = self.args
        paths = self.setup_resfiles()

        if config is not None:
            args = argparse.Namespace(**config)

        log_dir = paths['log']
        log_file = os.path.join(log_dir, f'{args.model}-{args.dataset}.log')

        # 移除所有之前配置的handlers，这是为了避免多进程情况之下重复日志输出
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # 获取logger并且设置日志级别
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        def _handler(logger, level, to_filepath = None):
            if to_filepath is not None:
                handler = logging.FileHandler(to_filepath)
            else:
                handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(level)
            logger.addHandler(handler)
            
        # 创建文件处理器来将日志信息写入文件，同时如何开启verbose则将日志打印终端
        _handler(logger, logging.INFO, to_filepath = log_file)
        _handler(logger, logging.INFO, to_filepath = None) if verbose else None

        return logger

    
    def __getattr__(self, name):
        if '_initialized' in self.__dict__ and self._initialized:
            return getattr(self.args, name)

    def __setattr__(self, name, value):
        if '_initialized' in self.__dict__ and self._initialized:
            setattr(self.args, name, value)
    
    def __str__(self) -> str:
        return str(self.args)
    

# 通用型模块
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    # 测试单例模式
    parser1 = ArgParser()
    parser2 = ArgParser()
    print(parser1 is parser2)