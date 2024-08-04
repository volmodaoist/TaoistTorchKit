import torch 
import numpy as np

import warnings
import copy as cp

from torchmetrics.metric import Metric
from torchmetrics import MetricTracker
from torchmetrics import MetricCollection

from torchmetrics import (Accuracy, Precision, Recall, F1Score, AUROC,
                          MeanSquaredError, 
                          MeanAbsoluteError)

 
class Tracker:
    def __init__(self, patience, delta = 0, task_type='multiclass', model=None, device=None, **kw):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.best_score = float('-inf')
        self.best_model = None
        self.best_epoch = None
        self.should_stop = False


        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device 
            
        self.metrics = self.task_metrics(task_type, **kw)
        self.tracker = MetricTracker(MetricCollection(self.metrics)).to(self.device)


    def increment(self):
        self.tracker.increment()
        
    def update(self, preds, reals):
        self.tracker.update(preds, reals)
        
    def compute(self, preds = None, reals = None):
        if preds is not None and reals is not None:
            self.update(preds, reals)
        return self.tracker.compute()
    
    def compute_all(self):
        return self.tracker.compute_all()
    
    def check_one_epoch(self, epoch, val_loss, preds = None, reals = None):
        score = -val_loss
        if score <= self.best_score - self.delta:
            if epoch - self.best_epoch > self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.update_best(epoch, val_loss)
        
        return self.compute(preds, reals)
        

    def update_best(self, epoch, val_loss):
        self.best_epoch = epoch
        self.best_model = cp.deepcopy(self.model)
        self.val_loss_min = val_loss

    def should_early_stop(self):
        return self.should_stop

    # NOTE 此时每个指标的名字必须对应 Logger 对象里面的字符串模版
    def task_metrics(self, task_type, **kw):
        if task_type == 'multiclass':
            num_classes = kw.get('num_classes', None)
            if num_classes is None:
                warnings.warn("num_classes not specified; defaulting to 10.")
                num_classes = 10
            metrics = {'Acc': Accuracy(num_classes=num_classes, task = task_type),
                       'F1':  F1Score(num_classes=num_classes, task = task_type),
                       'P':   Precision(num_classes=num_classes, task = task_type),
                       'Recall': Recall(num_classes=num_classes, task = task_type),
                       "AUROC": AUROC(num_classes=num_classes, task = task_type)}
                        
        elif task_type == 'rec':
            metrics = {'MAE': MeanAbsoluteError(),
                       'RMSE': MeanSquaredError(squared=False),
                       'NMAE': NormalizedMAE(),
                       'NRMSE': NormalizedMAE(), 
                       'Acc1': AccuracyWithinPercentage(1),
                       'Acc5': AccuracyWithinPercentage(5),
                       'Acc10': AccuracyWithinPercentage(10)}
        else:
            raise ValueError("Unsupported task type specified.")
        
        # 所有度量指标转入GPU
        for k,v in metrics.items():
            metrics[k] = v.to(self.device)
        
        return metrics
    
    

class NormalizedMAE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds, target):
        norm_factor = torch.abs(target).mean()
        self.total += torch.sum(torch.abs(preds - target) / norm_factor)
        self.count += target.numel()

    def compute(self):
        return self.total / self.count

class AccuracyWithinPercentage(Metric):
    def __init__(self, percentage):
        super().__init__()
        self.percentage = percentage
        self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds, target):
        tolerance = self.percentage / 100.0 * torch.abs(target)
        self.correct += torch.sum(torch.abs(preds - target) <= tolerance)
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total
