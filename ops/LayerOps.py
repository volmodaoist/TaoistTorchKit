# 这个模块包含了一些常用的模型组件，以及一些常用的工具
import torch.nn as nn



def remove_module(module: nn.Module, target_module_type: type) -> nn.Module:
    '''递归移除指定类型的模块，将其换成恒等映射，target_module_type 标记要被替换的模块
    '''
    for child_name, child_module in module.named_children():
        if isinstance(child_module, target_module_type):
            setattr(module, child_name, nn.Identity())
        else:
            remove_module(child_module, target_module_type)
    return module


def replace_A2B(module: nn.Module, source_module_type: type, target_module_creatfunc) -> nn.Module:
    '''递归找出所有类型A模块，就成换成另一类型B模块
       其中 target_module_creatfunc 是一个用于创建目标模块的可调用对象，接收源模块作为输入。
    '''
    for child_name, child_module in module.named_children():
        if isinstance(child_module, source_module_type):
            target_module = target_module_creatfunc(child_module)
            setattr(module, child_name, target_module)
        else:
            replace_A2B(child_module, source_module_type, target_module_creatfunc)
    return module



def remove_bn(module: nn.Module) -> nn.Module:
    ''' 递归移除BN模块，将其换成恒等映射，主要作用在于验证BN模块带来的性能提升，
        或是为了令模型适配差分隐私训练模式，当然 opacus 自带的  ModuleValidator.fix 更好!
    '''
    return remove_module(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))



def replace_bn2gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    ''' 递归替换所有 BN 模块将其变成 GN 模块，在差分隐私的上下文中，
        使用标准的批量归一化需要使用所有样本的统计信息，可能会泄露个体数据点的信息，
        因而通常会使用 Group Normalization 作为代替品，其对同层样本通道进行分组，分别计算每组的均值和方差。
    '''
    
    def _create_gn(bn_module):
        """从BN模块创建GN模块，保持通道数相同。"""
        num_channels = bn_module.num_features
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    
    return replace_A2B(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d), _create_gn)
