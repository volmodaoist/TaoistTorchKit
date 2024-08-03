import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import platform
     
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class TrainTools:
    @staticmethod
    def get_dataloaders(train_set, valid_set, test_set, batch_size=1024, num_workers=8, collate_fn=default_collate):
        '''
        description: 创建训练集、验证集、测试集的数据加载器
        example: 
            >>> train_loader, valid_loader, test_loader = TrainTools.setup_dataloaders(train_set, valid_set, test_set)
        '''
        train_loader = DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True, pin_memory=True, 
                                             num_workers=num_workers, collate_fn=collate_fn)
        
        valid_loader = DataLoader(valid_set, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, 
                                             num_workers=num_workers, collate_fn=collate_fn)
        
        test_loader = DataLoader(test_set, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True, 
                                             num_workers=num_workers, collate_fn=collate_fn)
        
        return train_loader, valid_loader, test_loader

    @staticmethod
    def get_lossfn(loss_name):
        '''
        description: Retrieves a loss function by its name.
        example: 
        >>> loss_fn = TrainTools.get_lossfn('mse')
        '''
        nick_name = loss_name.lower()
        loss_mapping = {
            'l1': nn.L1Loss,
            'smoothl1loss': nn.SmoothL1Loss,
            'mse': nn.MSELoss,
            'ce': nn.CrossEntropyLoss,
        }
        if nick_name not in loss_mapping:
            raise ValueError(f"Invalid loss name: {loss_name}")
        return loss_mapping[nick_name]()

    @staticmethod
    def get_optimizer(optimizer_name, parameters, lr, weight_decay, **kwargs):
        '''
        description: Retrieves an optimizer by its name and initializes it with the given parameters.
        example: 
        >>> optimizer = TrainTools.get_optimizer('adam', model.parameters(), lr=0.001, weight_decay=0.0001)
        '''
        nick_name = optimizer_name.lower()
        optimizer_mapping = {
            'sgd': optim.SGD,
            'adam': optim.Adam, 
            'adamw': optim.AdamW, 
            'adamax': optim.Adamax,
            'rmsprop': optim.RMSprop,
            'adagrad': optim.Adagrad,
            'adadelta': optim.Adadelta, 
        }
        if nick_name not in optimizer_mapping:
            raise ValueError(f"Invalid optimizer name: {optimizer_name}")
        opt = optimizer_mapping[nick_name]
        if nick_name != 'sgd':
            return opt(parameters, lr=lr, weight_decay=weight_decay)
        return opt(parameters, lr=lr, weight_decay=weight_decay, momentum=kwargs.get('momentum', 0.9))



    @staticmethod
    def seed_everything(seed):
        '''
        description: Sets the seed for reproducibility across random number generators.
        example: 
            >>> TrainTools.seed_everything(42)
        '''
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def server_details(verbose=True):
        '''
        description: Prints and returns details about the current computing platform.
        example: 
            >>> details = TrainTools.server_details()
        '''
        os_info = platform.platform()
        os_version = platform.version()
        system_name = platform.system()
        architecture = platform.architecture()[0]
        machine_type = platform.machine()
        node_name = platform.node()
        processor_type = platform.processor()

        description = (
            f"The current computing platform is running on {system_name} ("
            f"{os_info}) with version {os_version}. The system architecture is "
            f"{architecture} and the machine type is {machine_type}. The node name of the "
            f"computer is {node_name}, and it is powered by a {processor_type} processor."
        )

        if verbose:
            print(description)
        
        return description




def optimizer_zero_grad(*optimizers):
    '''
    description: Zeroes the gradients of the provided optimizers.
    example: 
        >>> TrainTools.optimizer_zero_grad(optimizer1, optimizer2)
    '''
    for optimizer in optimizers:
        optimizer.zero_grad()


def optimizer_step(*optimizers):
    '''
    description: Performs a step for each of the provided optimizers.
    example: 
        >>> TrainTools.optimizer_step(optimizer1, optimizer2)
    '''
    for optimizer in optimizers:
        optimizer.step()
    

def lr_scheduler_step(*lr_schedulers):
    '''
    description: Performs a step for each of the provided learning rate schedulers.
    example: 
        >>> TrainTools.lr_scheduler_step(scheduler1, scheduler2)
    '''
    for scheduler in lr_schedulers:
        scheduler.step()