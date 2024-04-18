import torch 
import torch.nn as nn

import os
import copy as cp

from torchmetrics.metric import Metric
from torchmetrics import MetricTracker
from torchmetrics import MetricCollection

class RunningLossSTD(Metric):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        self.add_state("running_loss", default = torch.Tensor([0.0]), dist_reduce_fx="sum")
        self.add_state("n_observations", default = torch.Tensor([0]), dist_reduce_fx="sum")
		# 添加状态，dist_reduce_fx 指定了用在多进程之间聚合状态所用的函数

    def update(self, preds, target):
        # 更新状态，这是一个标准损失函数，因而只有两个输入项
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
        
        self.best_loss = float('inf') if loss_fn else None
        self.best_model_state = model.state_dict() if model else None
        
        
        # 如果传入损失函数，则将损失函数记录在内
        if loss_fn and isinstance(loss_fn, nn.Module):
            if isinstance(metrics, list):   # 若是列表
                metrics = [RunningLossSTD(loss_fn)] + metrics
            elif isinstance(metrics, dict): # 若是字典
                metrics = {'RunningLoss': RunningLossSTD(loss_fn), **metrics}
            
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
    
    def compute(self, scalar = 1, return_log = False):
        lk = 'RunningLoss'
        res = {k: v.item() for k,v in self.tracker.compute().items()}
        
        if return_log:
            res2str = f"{lk}: {res[lk]:.4f}, " if self.loss_fn else f""\
                    + ", ".join([f"{k}: {v.item() * scalar:.2f}{str('%') if scalar == 100 else str()}" 
                                for k, v in res.items() if k != lk and v.numel() == 1])
            respair = (res, res2str)
        
        # 只有损失函数与模型均不为空的时候，才需要维护最优损失状态之下的模型权重
        if self.loss_fn is None or self.model is None:
            return respair if return_log else res
        
        if res[lk] < self.best_loss:
            self.best_loss = res[lk]
            self.best_model_state = cp.deepcopy(self.model.state_dict())
         
        return respair if return_log else res
    
    
    def compute_all(self):
        return self.tracker.compute_all()
    
    def checkpoint(self, filename = None,  persistent = True):
        res = {'Track': self.compute_all()}
        if self.model and self.loss_fn and self.optimizer:
            res = {
                'model': self.model,
                'best_model_state': self.best_model_state,
                'optimizer_state': self.optimizer.state_dict(),
                **res
            }
        
        if persistent and filename:
            torch.save(res, filename)
            
        return res    
    
    # 如果已有存储点，使用mkey 作为关键字合并当前结合存储点
    def merge_checkpoint(self, filename, mkey):
        return _merge_checkpoint(self.checkpoint(persistent = False), mkey, filename) 
    
    

class MultiTracker:
    def __init__(self, metrics, prefix = None, postfix = None, 
                                model = None,  loss_fn = None, optimizer = None, device = None):
        self.model = model
        self.loss_fn = loss_fn
        self.loss_cp = []
        self.optimizer = optimizer

        if not isinstance(metrics, list):
            metrics = [metrics]
                    
        self.best_loss = float('inf') if loss_fn else None
        self.best_model_state = model.state_dict() if model else None
        
        
        for i, mcomb in enumerate(metrics):
            if loss_fn and isinstance(loss_fn, nn.Module):
                mcomb = {'RunningLoss': RunningLossSTD(loss_fn), **mcomb}
        
            metrics[i] = MetricCollection(mcomb, prefix = prefix, postfix = postfix)
                
        # 使用组合而非继承的方式封装一个或多个 MetricTracker
        self.tracker = [MetricTracker(mc).to(device) for mc in metrics]
        

    def increment(self):
        for tk in self.tracker:
            tk.increment()
        
    def update(self, preds_lst, targets_lst):
        preds_lst = [preds_lst] if not isinstance(preds_lst, list) else preds_lst
        targets_lst = [targets_lst] if not isinstance(targets_lst, list) else targets_lst
        for i, (preds, targets) in enumerate(zip(preds_lst, targets_lst)):
            self.tracker[i].update(preds, targets)
    
    # 更新复杂的复合损失函数
    def update_losscp(self, loss):
        self.loss_cp += [loss]
        if self.loss_fn and  self.model and loss < self.best_loss:
            self.best_loss = loss
            self.best_model_state = cp.deepcopy(self.model.state_dict())
 
       
    def best_metric(self, return_step = True):
        return [tk.best_metric(return_step = return_step) 
                        for tk in self.tracker]
    
    
    def compute(self, scalar = 1, return_log = False):
        lk = 'RunningLoss'
        results, respairs = [{k:v.item() for k,v in tk.compute().items()} for tk in self.tracker], []
        
        if return_log:
            for res in results:
                res2str = f"{lk}: {res[lk]:.4f}, " if self.loss_fn else f""\
                        + ", ".join([f"{k}: {v.item() * scalar:.2f}{str('%') if scalar == 100 else str()}" 
                                    for k, v in res.items() if k != lk and v.numel() == 1])
                respairs += [(res, res2str)]
        
        return respairs if return_log else results
    
    
    def compute_all(self):
        return [tk.compute_all() for tk in self.tracker]
    
    
    def checkpoint(self, filename = None, persistent = True):
        res = {'Track': self.compute_all()}
        if self.model and self.loss_fn  and self.optimizer:
            res = {
                'model': self.model,
                'loss_cp': self.loss_cp,
                'best_model_state': self.best_model_state,
                'optimizer_state': self.optimizer.state_dict(),
                **res
            }
           
        if persistent and filename:
            torch.save(res, filename)
            
        return res

    # 如果已有存储点，使用mkey 作为关键字合并当前结合存储点
    def merge_checkpoint(self, filename, mkey):
        return _merge_checkpoint(self.checkpoint(persistent = False), mkey, filename) 
        
    

def _merge_checkpoint(checkpoint, mkey, filename):
    if not os.path.exists(filename):
        merged_res = {mkey: checkpoint}
    else:
        merged_res = torch.load(filename)
        merged_res[mkey] = checkpoint
    
    torch.save(merged_res, filename)
    return merged_res 
    