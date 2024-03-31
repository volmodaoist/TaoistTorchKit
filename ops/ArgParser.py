import torch
import argparse
import numpy as np
import os
import random




GPU = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CPU = torch.device("cpu")

# 任务超参数设置模块
class ArgParser:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if self.__initialized:
            return
                    
        self.parser = argparse.ArgumentParser()
         
        # 通用性参数
        self.parser.add_argument("--model", default = None, type = str,              # 模型
                                help = "The model needed for the specific task.")
        self.parser.add_argument("--backbone", default = None, type = str,           # 基座模型
                                help = "The backbone for the specific model.")
        self.parser.add_argument("--dataset", default = None, type = str,            # 数据集
                                help = "Dataset for train/test model.")
        self.parser.add_argument("--seed", default = 3407, type = int,               # 随机数设置
                                 help = "Random seed for initializing training.")
        self.parser.add_argument("--mode", default = 10, type = int,                  # 设置训练模型和测试模块
                                 help = "Setting for test (0), train (1), or both (10) model, ")
        self.parser.add_argument("--device", default = GPU, type = torch.device,     # 模式:使用GPU或CPU
                                 help = "Select GPU/CPU mode.")
        
        '''
        基本参数设置:
         - 主要是模型训练阶段的优化器、调度器参数,
         - 其次是数据集读入参数，包括读取线程、数据尺寸,
        '''
        
        # 设置学习率
        self.parser.add_argument("-lr", "--learning_rate", default = 0.01, type = float,
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
        self.parser.add_argument("-pl", "--pairloss", default = 'L2', type = str,
                                 help="Pairloss for adversarial logit pairing.")
         
        # 分类任务: 设置分类模型需要分类的个数
        self.parser.add_argument("--num-classes", default = 10, type = int,
                    help = "The number of classes the model needs to classify.") 
        
        
        # 重构任务: 编码解码结构的中间维度
        self.parser.add_argument('-dim', "--dim", default = 10, type = int,
                    help = "The dimension of latent space.")
        
        self.args = self.parser.parse_args()
        self.__initialized = True
    

    def parse_args(self, seed:int = None):            
        seed_everything(seed if seed else self.args.seed)
        return self.args
    
    def update_args(self, config:dict = None):
        '''此处使用 vars 获取命令行参数命名空间的字典引言，更新这个字典，
           命令参数由于引用性质也会随之更新, 如此 ArgParser 对象的可以提高通用性!
        '''
        if config is not None:
            vars(self.args).update(config)
        
        seed_everything(config.get('seed', self.args.seed))
        return self.args
    
    def __getattr__(self, name):
        return getattr(self.args, name)


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