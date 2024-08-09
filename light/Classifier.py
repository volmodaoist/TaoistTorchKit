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
                
    
  
    