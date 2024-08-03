import torch
import torch.nn as nn
import torch.optim as optim
import sys

import time


sys.path.append('./')
sys.path.append('../')

from tqdm import tqdm
from utils import (
    Tracker,
    TrainTools,
    optimizer_step, 
    optimizer_zero_grad
)


class Classifier(nn.Module):
    def __init__(self, model, patience = 10, delta = 0, task = 'multiclass'):
        super().__init__()
        self.model = model
        self.tracker = Tracker(patience = patience, delta = delta, task_type = task, 
                              model = self)

                    
    def forward(self, x):
        return self.model(x)

    def configure_opt(self, args):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.cuda()
        self.loss_fn = TrainTools.get_lossfn(args.loss_func)
        self.optimizer = TrainTools.get_optimizer(args.optim, self.parameters(), args.lr, args.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', 
                                                              factor = args.lr_decay_gamma, 
                                                              patience = args.lr_decay_step, 
                                                              threshold=0.01)

    def train_one_epoch(self, dataloder):
        self.train()
        t1 = time.time()
        torch.set_grad_enabled(True)
        for inputs, labels in tqdm(dataloder):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            preds = self.forward(inputs)
            loss = self.loss_fn(preds, labels)
            loss.backward()
            
            optimizer_step(self.optimizer)
            optimizer_zero_grad(self.optimizer)
        
        torch.set_grad_enabled(False)
        t2 = time.time()
        self.eval()
        return loss.item(), t2 - t1

    
    def eval_one_epoch(self, data_loader):
        val_loss = 0
        preds, reals = [], []
        for batch in tqdm(data_loader):
            inputs, value = batch
            inputs, value = inputs.to(self.args.device), value.to(self.args.device)
            pred = self.forward(inputs)
            val_loss += self.loss_fn(pred, value).item()
            preds.append(pred)
            reals.append(value)
        preds = torch.cat(preds, dim=0)
        reals = torch.cat(reals, dim=0)
        self.scheduler.step(val_loss)
        return val_loss, preds, reals
    
    
    def test_one_epoch(self, dataModule):
        _, preds, reals = self.eval_one_epoch(dataModule.test_loader)
        
        return self.compute(preds, reals)

    def valid_one_epoch(self, epoch, dataModule):
        val_loss, preds, reals = self.eval_one_epoch(dataModule.valid_loader)
        self.tracker.check_stop(epoch, val_loss)
        self.tracker.increment()
        return self.tracker.compute(preds, reals)
    
    
  

    