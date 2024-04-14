import torch 
import torch.nn as nn

import copy as cp

from torchmetrics.metric import Metric
from torchmetrics import MetricTracker
from torchmetrics import MetricCollection

class RunningLoss(Metric):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.add_state("running_loss", default = torch.Tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("n_observations", default = torch.Tensor([0]), dist_reduce_fx="sum")
		# 添加状态，dist_reduce_fx 指定了用在多进程之间聚合状态所用的函数

    def update(self, preds, target):
        # 更新状态
        self.running_loss += self.criterion(preds, target)
        self.n_observations += 1

    def compute(self):
        return self.running_loss / self.n_observations
    
    

class Tracker:
    def __init__(self, metrics, prefix = None, postfix = None, 
                                model = None,  loss_fn = None, optimizer = None, device = None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        self.best_loss = float('inf')
        self.best_model_state = model.state_dict()
        
        # 如果传入损失函数，则将损失函数记录在内
        if loss_fn and isinstance(loss_fn, nn.Module):
            # 若是列表
            if isinstance(metrics, list):
                metrics = [RunningLoss(loss_fn)] + metrics
            # 若是字典
            elif isinstance(metrics, dict):
                metrics = {'RunningLoss': RunningLoss(loss_fn), **metrics}
            
    
        # 多组指标打包在一起
        if not isinstance(metrics, MetricCollection):
            metrics = MetricCollection(metrics, prefix = prefix, postfix = postfix)
        
        # 使用组合而非继承的方式封装 MetricTracker
        self.tracker = MetricTracker(metrics).to(device)

    def increment(self):
        self.tracker.increment()
        
    def update(self, preds, labels):
        self.tracker.update(preds, labels)
        
    def best_metric(self, return_step = True):
        return self.tracker.best_metric(return_step = return_step)
    
    def compute(self, return_log = False):
        res = self.tracker.compute()

        if self.loss_fn is None or self.model is None:
            return res
        
        if res['RunningLoss'] < self.best_loss:
            self.best_loss = res['RunningLoss']
            self.best_model_state = cp.deepcopy(self.model.state_dict())
        
        if return_log:
            res2str = ", ".join([f"{k}: {v:.2f}" 
                                 for k, v in res.items()])
            res = (res, res2str)
        
        return res
    
    def compute_all(self):
        return self.tracker.compute_all()
    
    def checkpoint(self, filename):
        res = {'Track': self.compute_all()}
        if self.model and self.loss_fn  and self.optimizer:
            res = {
                'model': self.model,
                'best_model_state': self.best_model_state,
                'optimizer_state': self.optimizer.state_dict(),
                **res
            }
        
        torch.save(res, filename)
        return res
        
        
        
        
        