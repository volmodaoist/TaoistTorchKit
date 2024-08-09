import argparse
import json
import yaml

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
                                help = "The model needed for the specific task")
        self.parser.add_argument("-A", "--activate", default = 'relu', type = str,   # 使用什么激活函数
                                 help = "Activata-function for model.")
        self.parser.add_argument("--input-size", nargs = 4, type = int,              # 模型输入尺寸
                            help = "Set channel for images that the model reads in batches.")
        self.parser.add_argument("--load-path", default = None, type = str,          # 预训练权重路径
                    help = "Load the pre-trained model for evaluation.")
        self.parser.add_argument("--dataset", default = None, type = str,            # 数据集
                                help = "Dataset for train/test model.")
        self.parser.add_argument("--valid-size", default=0.1, type=float,            # 验证集的比例
                                 help="Valid set proportion.")
        self.parser.add_argument("--test-size", default=0.1, type=float,             # 测试集的比例
                                 help="Test set proportion.")
        self.parser.add_argument("--seed", default = 3407, type = int,               # 随机数设置
                                 help = "Random seed for initializing training.")
        self.parser.add_argument("--mode", default = "train/test", type = str,       # 使用训练或测试模型
                                 help = "Select Train/Test mode")
        
        # 针对分类任务的特定参数
        self.parser.add_argument("-aug", "--aug-mask", default="000000", type=str,
                                 help="Mask used to specify data augmentation")     # 数据增强的方案掩码
        self.parser.add_argument("-at", "--attack-type", default=None, type=str,
                                 help="Attack for adversarial train/test.")         # 对抗训练使用的攻击类型
        self.parser.add_argument("-an", "--attack-norm", default=None, type=str,
                                 help="Attack for adversarial train/test.")         # 对抗训练使用的攻击约束
        self.parser.add_argument("-ag", "--aggr", default = "gcn", type=str, 
                                 help = "Aggregate function of GNN.")               # 聚合函数
        self.parser.add_argument("--num-classes", default = 10, type = int,          
                                 help = "The number of classes for classifier.")    # 分类需要区分的类别个数
        self.parser.add_argument("--dim", default = 10, type = int,
                                 help = "The dimension of latent space.")           # 编码器的中间维度
        self.parser.add_argument("--gnn", default = "gcn", type=str, 
                                 help = "Auxiliary GNN model.")                     # 辅助其它任务的图神经网络
        self.parser.add_argument("--backbone", default = None, type = str,                           
                                 help = "The backbone model for specific task.")    # 特定任务的骨干网络
        self.parser.add_argument("--topk", default = None, type = float,                           
                                 help = "Topk for something e.g. dataset/acc.")     # 选取 TopK 排名的某物
        self.parser.add_argument("--mark", default = 1, type = int,                           
                                 help = "Version Mark for something.")              # 选取某物的版本号
     

        # 训练模型的核心模块+基本参数: 损失函数/优化器/早停; 学习率/批度/代际/轮次/激活函数
        self.parser.add_argument("--loss-func", default=None, type=str,
                                 help="Loss function for training the model")
        self.parser.add_argument("--optim", default=None, type=str,
                                 help="Optimizer for training the model")
        self.parser.add_argument("--patience", default=10, type=int,
                                 help="Number of epochs with no improvement after which training will be stopped.")
        self.parser.add_argument("--lr", default = 0.01, type = float,
                                 help = "Epoch for train model.")
        self.parser.add_argument("--lr-decay-step", default = 10, type = int,
                            help = "Used to decide how many steps to decay the learning rate every.")
        self.parser.add_argument("--lr-decay-gamma", default = 0.5, type = float,
                            help = "Used to decide how much the learning rate decays each time.")
        self.parser.add_argument("--batch-size", default = 128, type = int,
                                 help = "Batch size for train/test model.")
        self.parser.add_argument("--num-workers", default = 8, type = int,
                            help = "Number of threads to loading dataset.")
        self.parser.add_argument("--epochs", default = 10, type = int,
                                 help = "Epoch for train model.")
        self.parser.add_argument("--rounds", default = 3,  type = int,
                                 help = "Rounds for experiment.")
        self.parser.add_argument("--weight-decay",  default = 1e-5, type = float, 
                                 help = "Used to decide weight decay in optimizer.")
         
         
        
        self.__initialized = True

    def parse_args(self):
        return self.parser.parse_args()
    
    def update_from_json(self, json_file):
        with open(json_file, 'r') as f:
            json_data = json.load(f)
            for key, value in json_data.items():
                if key in self.parser._option_string_actions:
                    self.parser._option_string_actions[key].default = value

    def update_from_yaml(self, yaml_file):
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
            for key, value in yaml_data.items():
                if key in self.parser._option_string_actions:
                    self.parser._option_string_actions[key].default = value
