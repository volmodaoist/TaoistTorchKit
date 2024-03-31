import numpy as np
import matplotlib.pyplot as plt

FMTS = (
        'r-s',   # 红色实线方形标记
        'g--o',  # 绿色虚线实线圆形标记
        'b-.^',  # 蓝色点划线三角形向上标记
        'c:v',   # 青色点线三角形向下标记
        'y-s',   # 黄色实线方形标记
        'm-o',   # 洋红色实线圆形标记
        'k-^',   # 黑色实线三角形向上标记
        'b-v',   # 蓝色实线三角形向下标记
        'g-*',   # 绿色实线星形标记
        'r-p',   # 红色实线五角形标记
        'c-h',   # 青色实线六边形标记
        'm-D',   # 洋红色实线菱形标记
        'y-x',   # 黄色实线叉形标记
        'k-|',   # 黑色实线竖直线形标记
        'b-_',   # 蓝色实线水平线形标记
    )

# 辅助函数
def plot_figure(X, Y = None, xlabel = None, ylabel = None, legend = None, 
                xlim = None, ylim = None, xscale = 'linear', yscale = 'linear',
                fmts = FMTS, figsize = (3.5, 2.5), axes = None, save_name:str = 'tmp.png'):
    
    def set_figsize(figsize=(3.5, 2.5)):
        plt.rcParams['figure.figsize'] = figsize

    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()
    
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    def has_one_axis(X):
        # 检查传入的数据是不是一个 numpy 数组且只有一个维度，或者X是不是一个列表，且列表元素不存在序列长度 len
        return (X is not None) and (hasattr(X, "ndim") and X.ndim == 1) or (isinstance(X, list)
                and not hasattr(X[0], "__len__"))


    X = [X] if has_one_axis(X) else X
    Y = [Y] if has_one_axis(Y) else Y
    if Y is None:       
        X, Y = [[]] * len(X), X

     # 我们允许X长度小于Y, 我们通过重复X来实现多个曲线共享同一个横轴，但是仅当 len(X) = 1, len(Y) > 1 这种情况是有意义的
    if len(X) != len(Y):
        X = X * len(Y)
        
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.savefig(save_name)


