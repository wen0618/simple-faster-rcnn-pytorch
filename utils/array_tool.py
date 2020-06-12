"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data): #将tensor数据转化为numpy
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()
#将变量从计算图中分离（使得数据独立，以后你再如何操作都不会对图，对模型产生影响），
#如果是gpu类型变成cpu的（cpu类型调用cpu方法没有影响），再转化为numpy数组
#Returns a new Variable, detached from the current graph。
#将某个node变成不需要梯度的Varibale。
#因此当反向传播经过这个node时，梯度就不会从这个node往前面传播。

def totensor(data, cuda=True):   #/将数据转化为cuda或者tensor类型   cuda=True表示转化为cuda类型
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()    #隔离变量
    if cuda:
        tensor = tensor.cuda()
    return tensor


def scalar(data): #取出数据的值
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]    #如果是numpy类型（必须为1个数据 几维都行） 取出这个数据的值    
    if isinstance(data, t.Tensor):
        return data.item() # //如果是tensor类型 调用pytorch常用的item方法 取出tensor的值
