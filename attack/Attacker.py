import torch
import numpy as np
import foolbox as fb

from tqdm import tqdm
from fractions import Fraction

from torchmetrics import (Accuracy, Precision, Recall, F1Score, AUROC)
from torchmetrics import MetricTracker
from torchmetrics import MetricCollection


class ClassifierAttackerFb:
    def __init__(self, model, num_classes, task_type = 'multiclass', 
                        bounds = (0, 1), 
                        preprocessing = None):
        
        self.model = model
        self.bounds = bounds
        self.preprocessing = preprocessing   
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        
        self.fmodel = fb.PyTorchModel(
            model = self.model, 
            bounds = self.bounds, 
            device = self.device,
            preprocessing = self.preprocessing
        )
        
        metrics = {
            'Acc': Accuracy(num_classes=num_classes, task = task_type),
            'AUROC': AUROC(num_classes=num_classes, task = task_type)
        }
        self.tracker = MetricTracker(MetricCollection(metrics)).to(self.device)
        
    def config_attacker(self, attacks):
        self.attacks = attacks
        return self
        
      
    def attack(self, dataloader, epsilons):
        ans = {}
        for attack_key, attack_v in self.attacks.items():
            for eps in epsilons:
                self.tracker.increment()
                for inputs, labels in tqdm(dataloader, desc = f'AT: {attack_key}, Eps: {Fraction(eps).limit_denominator()}'):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    _, clipped, is_adv = attack_v(self.fmodel, inputs, labels, epsilons = [eps])
                    preds = self.model(clipped[0])
                    self.tracker.update(preds, labels)
            
            results = self.tracker.compute_all()
            ans[attack_key] = results
        return ans
                
            
    
    
    
    
# 以下字典主要用于提醒
ATTACK_POOL_Linf = {
    'fgsm': fb.attacks.LinfFastGradientAttack,
    'pgd': fb.attacks.LinfPGD, 
    'apgd': fb.attacks.LinfAdamPGD,
    'deepfool': fb.attacks.LinfDeepFoolAttack,
}

ATTACK_POOL_L2 = {
    'fgsm': fb.attacks.L2FastGradientAttack,
    'pgd': fb.attacks.L2ProjectedGradientDescentAttack, 
    'apgd': fb.attacks.L2AdamPGD,
    'cw': fb.attacks.L2CarliniWagnerAttack,
    'deepfool': fb.attacks.L2DeepFoolAttack,
}

ATTACK_POOL_L1 = {
    'fgsm': fb.attacks.L1FastGradientAttack,
    'pgd': fb.attacks. L1ProjectedGradientDescentAttack, 
    'apgd': fb.attacks.L1AdamPGD,
    'deepfool': fb.attacks.L2DeepFoolAttack,
}