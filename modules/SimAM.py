import torch
import torch.nn as nn

'''
文章描述: 一种无参数的即插即用注意力，可参考文章提到的其它注意力模块
参考链接: http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
'''
class SimAM(nn.Module):
    def __init__(self, e_lambda = 1e-4):
        super(SimAM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
    
    def forward(self, x):
        _ , _, h, w = x.shape
        n = w * h - 1
        z = (x - x.mean(dim = [2, 3], keepdim = True)).pow(2)
        y = z / (4 * (z.sum(dim = [2, 3], keepdim = True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)
    

if __name__ == '__main__':
    r = torch.randn(3, 3, 32, 32)
    am = SimAM()
    print(am(r).shape)