import torch
import torch.nn as nn
import torch.nn.functional as F


'''
来自于论文: Feature Denoising for Improving Adversarial Robustness
参考仓库: 
  - https://github.com/A-LinCui/DenoisingNet_Adversarial_Training/blob/master/Denoising_ResNet.py
  - https://github.com/Aiqz/Edge-Enhancement/blob/c9c861351fe516cdedcb701cfb2c2ddbbf91146a/ImageNet/models_imagenet/resnet_fd.py#L105
'''
class NonLocalDenoise(nn.Module):
    def __init__(self, channels, embed = True, softmax = True):
        super(NonLocalDenoise, self).__init__()
        self.embed = embed
        self.softmax = softmax
        
        # 沿着通道维度进行嵌入        
        self.embed_channels = channels // 2 if embed else channels 
        self.theta = nn.Conv2d(channels, self.embed_channels, kernel_size = 1, stride = 1, bias=False) if embed else nn.Identity()
        self.phi   = nn.Conv2d(channels, self.embed_channels, kernel_size = 1, stride = 1, bias=False) if embed else nn.Identity()
        
            
    def forward(self, x):
        n, c, h, w = x.size()
        
        theta_x = self.theta(x).view(n, self.embed_channels, -1)
        phi_x = self.phi(x).view(n, self.embed_channels, -1)
            
        g_x = x.view(n, c, -1)
        f = torch.matmul(theta_x.permute(0, 2, 1), phi_x)
        
        if self.softmax is True:
            f = F.softmax(f, dim = -1)
            
        y = torch.matmul(f, g_x.permute(0, 2, 1))
        y = y.permute(0, 2, 1).contiguous().view(n, c, h, w)

        return x + y