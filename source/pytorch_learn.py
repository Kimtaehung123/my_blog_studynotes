import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.random import set_rng_state # 
# pytorch框架使用的数据类型是tensor,要把输入数据转化为tensor
# 把数据从numpy变为tensor
X = torch.from_numpy(x)
Y = torch.from_numpy(y)

model = nn.Linear(1,1)  # out = w * input + b
loss_fn = nn.MSELoss() # loss function
opt = torch.optim.SGD(model.parameters(),lr = 0.0001) # 梯度下降有很多方向，随机指的就是随机选取一个方向

for epoch in range(5000):
    for x,y in zip(X,Y):
        y_pred = model(x)          # 使用模型预测
        loss = loss_fn(y,y_pred)   # 根据预测结果计算损失
        opt.zero_grad()            # 把变量梯度清0
        loss.backward()            # 求解梯度
        opt.step()                 # 参数更新；单个优化步

# 因为是最简单的linear模型，可以直接使用weights得到权值
model.weight

model.bias

# plt画图需要使用numpy类型，把tensor转化为numpy
# model(X)模型预测输出值也是个tensor，但是包括梯度等，使用.data得到预测结果的值
plt.plot(X.numpy(),model(X).data.numpy())

#%%[markdown]
# 张量与基本数据类型
# pytorch最基本的操作对象是Tensor（张量）；
# 表示一个多维矩阵；
# 类似于NumPy的ndarrays;
# 张量可以在GPU上使用以加速计算

#%%
# 创建tensor
import torch 
x = torch.rand(2,3) # 均匀分布的随机数
print(x)
# %%
x = torch.randn(3,4) # 正态分布随机数
print(x)

#%%
x = torch.zeros(2,3)
print(x)

#%%
x = torch.ones(2,3,4)
print(x)

#%%
print(x.size()) # 返回张量尺寸，可以选择要第几维的尺寸
print(x.shape) # 不可以选择要第几维度的尺寸
print(x.size(0))

#%%[markdown]
# Tensor最基本的数据类型
# 32位浮点型：torch.float32
# 64位浮点型：torch.float64
# 32位整型：torch.int32
# 16位整型：torch.int16
# 64位整型：torch.int64
# 有的代码只写了float或者int，不建议这种不明确的写法
# Numpy转为Tensor：torch.from_numpy(numpy 矩阵)
# Tensor转为Numpy：Tensor矩阵.numpy()
#%%
x = torch.tensor([6,2],dtype=torch.float32) # 从列表中直接创建tensor
print(x) # output：tensor([6., 2.])，其中的点代表是一个float类型
print(x.type()) # output:torch.FloatTensor
# 数据类型转换
print(x.type(torch.int64))

#%%
import numpy as np
a = np.random.randn(2,3)
x1 = torch.from_numpy(a)
print(x1)
print(x1.numpy())

#%%[markdown]
# 张量运算与numpy一致，对应元素加减，一样的广播机制

#%%
x2 = torch.rand(2,3)
print(x1,x2)
print(x1+3)
print(x1.add(x2)) # x1不会改变
print(x1.add_(x2)) # !!加下划线代表元素会被改变

# 数组变形
print(x1.view(3,2)) # 这个view相当于numpy中的reshape
print(x1.mean())
x = x1.sum()
print(x)  # 一个值的向量
print(x.item()) # item用于返回一个标量值,只适用于tensor只有一个值的情况

#%%[markdown]
# 张量的自动微分(梯度计算)
# 将Torch.Tensor属性.requires_grad设置为True，
# pytorch将开始跟踪对此张量的所有操作。
# 完成计算后，可以调用.backward()并自动计算所有的梯度
# 该张量的梯度将累加到.grad属性中。

#%%
x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.requires_grad)

# Tensor数据结构的特点
# 分为三部分
# 第一部分：data，记录tensor的原有值
# 第二部分：grad,代表梯度，设置requires_grad属性，使用.backward()计算的梯度就存放在这里
# 第三部分：grad_fn,梯度函数

print(x.data)
print(x.grad) # 没有的话返回空，即None
print(x.grad_fn) # 没有的话返回空，即None

y = x + 2
print(y)
print(y.grad_fn) # <AddBackward0 object at 0x0000029A6249DB48> 说明是使用+的方法得到的

z = y * y + 3
out = z.mean()
print(z)
print(out)

out.backward() # d(out)/dx
print(x.grad)

#%%
# 当不需要梯度计算时,可以包含在如下的上下文环境中
with torch.no_grad():
    print((x ** 2).requires_grad)

#%%
# detach方法等价于with torch.no_grad():
y = x.detach() # 只包含x的值，不包含requires_grad
print(y)
print(y.requires_grad)

#%%
a = torch.tensor([2,4],dtype=torch.float32) # 可能会报错只有浮点类型可以要求梯度，因此修改这里的数据属性
print(a)
print(a.requires_grad)
print(a.requires_grad_(True)) # 使用下划线直接改变tensor属性

#%%
# 分解写法(torch的灵活性)
w = torch.randn(1,requires_grad=True)
b = torch.zeros(1,requires_grad=True)
learning_rate = 0.0001
for epoch in range(5000):
    for x,y in zip(X,Y):
        y_pred = torch.matmul(x,w) + b 
        loss = (y-y_pred).pow(2).mean()
        if not w.grad is None:
            w.grad.data.zero_() # 把梯度置为0
        if not b.grad is None:
            b.grad.data.zero_() # 把梯度置为0
               
        loss.backward() # 计算梯度
        with torch.no_grad():
            w.data -= w.grad.data * learning_rate
            b.data -= b.grad.data * learning_rate

#%%[markdown]
# 线性回归预测的是一个连续值；逻辑回归给出的是“是”和“否”的回答
# 逻辑回归使用sigmoid函数，将连续的变量映射到【0，1】之间的概率值；
# sigmoid函数是一个概率分布函数，给定某个输入，它将输出为一个概率值；

# 逻辑回归损失函数：
# 平方差所惩罚的是与损失为同一数量级的情形；
# 对于分类问题，最好使用交叉熵损失函数会更有效；交叉熵会输出一个更大的“损失”

# 交叉熵损失函数
# 交叉熵刻画的是实际输出（概率）与期望输出（概率）的距离，
# 即交叉熵的值越小，两个概率分布就越接近。
# 
# pytorch二元交叉熵损失----nn.BCELoss() 
# %%
from torch import nn
model = nn.Sequential(
    nn.Linear(15,1),
    nn.Sigmoid()
)
# print(model)
loss = nn.BCELoss()
opt = torch.optim.Adam(model.parameters(),lr= 0.0001)
# 模型对异常值敏感，如果只使用一个样本，而不是一个batch，模型的损失会剧烈振荡；
batches = 16
no_of_batch = 653/16

epochs = 1000
for epoch in range(epochs):
    for i in range(no_of_batch):
        start = i * batches
        end = start + batches 
        x = X[start:end]
        y = Y[start:end]
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        opt.zero_grad() # 梯度全部置为0
        loss.backward() # 计算损失的反向传播
        opt.step() # 执行单个优化步，参数更新

model.state_dict() # 变量词典
# astype是numpy的转换数据方法
((model(X).data.numpy() > 0.5).astype('int') == Y.numpy()).mean() 

#%%[markdown]
# 单层神经元要求数据必须是线性可分的；
# 异或问题无法找到一条直线分割两个类；
# 激活函数带来了非线性，使得模型的拟合能力大大增强；
# 多层感知机 + 如何使用pytorch内置的方法处理批次数据

#%%
# 假设X和Y都是torch向量，并且要转化为正确的数据类型
# torch.float32等价于torch.FloatTensor
# 创建模型
# 使用nn.Module自定义模型；
# 两个方法需要自己写
# __init__:初始化所有的层
# forward：定义模型的运算过程（前向传播过程）
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(20,64)
        self.linear_2 = nn.Linear(64,64)
        self.linear_3 = nn.Linear(64,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self,input):
        x = self.linear_1(input)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.sigmoid(x)
        return x

model = Model()

#%%
# 改写模型
import torch.nn.functional as F 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(20,64)
        self.linear_2 = nn.Linear(64,64)
        self.linear_3 = nn.Linear(64,1)
       
    def forward(self,input):
        x = self.linear_1(input)
        x = F.relu(x)
        x = self.linear_2(x)
        x = F.relu(x)
        x = self.linear_3(x)
        x = F.sigmoid(x) # 这些激活函数没有可训练参数
        return x

model = Model()
print(model)

lr = 0.0001
def get_model():
    model = Model()
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    return model,opt 

model,optim = get_model()

#%%
# 定义损失函数
loss_fn = nn.BCELoss()
batch = 64
no_of_batches = len(data)/batch 
epochs = 100
for epoch in range(epochs):
    for i in range(no_of_batch):
        start = i * batch
        end = start + batch
        x = X[start:end]
        y = Y[start:end]
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad(): # 不需要梯度监督
        print("epoch:",epoch,'loss:',loss_fn(model(X),Y).data.item())

#%%[markdown]
# 使用TensorDataset和DataLoader加载模型数据
# 使用dataset类进行重构，包装张量
# 对输入和输出切片只需要写一遍，不需要写两遍
#%%
from torch.utils.data import TensorDataset
HRdataset = TensorDataset(X,Y)

# 查看长度
print(len(HRdataset))
# 使用HRdataset可以直接对X和Y切片
model,optim = get_model()
for epoch in range(epochs):
    for i in range(no_of_batch):
        x,y = HRdataset[i*batch:i*batch+batch] 
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad(): # 不需要梯度监督
        print("epoch:",epoch,'loss:',loss_fn(model(X),Y).data.item())

# 以上批次获取方法是按照顺序获取的，模型可能学到顺序信息，怎么可以乱序
# 并且还是需要切片的方式取出批次，可以不用切片就可以取出吗？
# 可以使用dataloader类解决上述问题

from torch.utils.data import DataLoader
HR_ds = TensorDataset(X,Y)
HR_dl = DataLoader(HR_ds,batch_size=batch,shuffle=True)
model,optim = get_model()
for epoch in range(epochs):
    for x,y in HR_dl:
        y_pred = model(x)
        loss = loss_fn(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad(): # 不需要梯度监督
        print("epoch:",epoch,'loss:',loss_fn(model(X),Y).data.item())

#%%[markdown]
# 怎么打印准确率
# 怎么加入验证数据集