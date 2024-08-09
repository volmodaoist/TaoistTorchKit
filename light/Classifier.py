import torch
import torch.nn as nn
import torch.optim as optim
import sys

sys.path.extend(['./', '../', './TaoistTorchKit'])


from tqdm import tqdm
from utils import (
    TrainTools,
    optimizer_step, 
    optimizer_zero_grad
)

# 图像分类器模版，这个模版尚未支持对抗训练
class Classifier(nn.Module):
    def __init__(self, model, dataModule, data_parser = lambda x: (x,)):
        super().__init__()
        self.model = model
        self.dataM = dataModule
        self.data_parser = data_parser
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.to(self.device)
             
                    
    def forward(self, x):
        return self.model(*self.data_parser(x))

    def configure_opt(self, args):
        self.loss_fn = TrainTools.get_lossfn(args.loss_func)
        self.optimizer = TrainTools.get_optimizer(args.optim, self.parameters(), args.lr, args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', 
                                                              factor = args.lr_decay_gamma, 
                                                              patience = args.lr_decay_step, 
                                                              threshold = 0.01)
        return self

    def fit(self, epochs):
        for epoch in range(epochs):
            epoch_loss, time_cost = self.train_one_epoch()
            val_loss, preds, reals = self.valid_one_epoch()
            print(f'[{epoch + 1}/{epochs}]: loss: {epoch_loss:.4f}, vloss: {val_loss:.4f}, time cost: {time_cost:.4f}s')
        
        # NOTE 末次调用验证方法会导致 train_eval_time 装饰器里面的上下文管理器关闭梯度, 
        # NOTE 因而此时必须手动开启, 堪比内存管理一般的易错之处!
        
        torch.set_grad_enabled(True) 
        return self
        
    
    @TrainTools.train_eval_time
    def train_one_epoch(self, tracker = None):
        if tracker is not None:
            tracker.increment()
        avg_loss = 0
        for inputs, labels in tqdm(self.dataM.train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            scores = self.forward(inputs)
            
            if tracker is not None:
                tracker.update(scores, labels)    

            loss = self.loss_fn(scores, labels)
            loss.backward()
            avg_loss += loss.item()
            
            optimizer_step(self.optimizer)
            optimizer_zero_grad(self.optimizer)
        
        avg_loss /= len(self.dataM.train_loader)
        return avg_loss,

    
    def eval_one_epoch(self, data_loader):
        val_loss, preds, reals = 0, [], []
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            scores = self.forward(inputs)
            val_loss += self.loss_fn(scores, labels).item()
            
            preds.append(scores)
            reals.append(labels)
            
        val_loss /= len(data_loader)
        preds = torch.cat(preds, dim=0)
        reals = torch.cat(reals, dim=0)
        self.scheduler.step(val_loss)
        return val_loss, preds, reals
    
            
    def valid_one_epoch(self):
        return self.eval_one_epoch(self.dataM.valid_loader)
    
    def test_one_epoch(self):
        return self.eval_one_epoch(self.dataM.test_loader)
class Predictor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        y = self.mlp(x)
        return y


