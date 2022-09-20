#%%[markdown]
# binary crossentropy(二值交叉熵)
#%%
# 导入包，定义基本函数
# from_logits = True,先进行sigmoid处理
from random import sample
from sklearn import metrics
from sympy import true
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np 

def sigmoid(logits):
    return 1/(1+ np.exp(-logits))

def crossentropy(y_true,y_pred):
    r = -(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return np.average(r,axis = -1)
#%%
# test1
y_true = [0, 1, 0, 0]
y_pred = [-18.6, 0.51, 2.94, -12.8]

# 调用类方法
jg = BinaryCrossentropy(from_logits=True)(y_true,y_pred).numpy()
print(jg)

# 使用公式计算
y_pred = np.array(y_pred)
y_prob = sigmoid(y_pred)
print(y_prob)
losses = 0
for i in range(len(y_true)):
    losses += -(y_true[i] * np.log(y_prob[i]) + (1-y_true[i]) * np.log((1-y_prob[i])))

losses /= 4 # 取均值
print(losses)

# 实验结果正确
# %%
# 使用不同的reduction方法
# test2
y_true = [[0, 1], [0, 0]]
y_pred = [[-18.6, 0.51], [2.94, -12.8]]
y_pred = np.array(y_pred)
y_true = np.array(y_true)

y_prob = sigmoid(y_pred)

# 调用函数
jg = BinaryCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.SUM)(y_true,y_pred).numpy()
print(jg)

# 公式计算
r = crossentropy(y_true,y_prob)
print(r)
print(sum(r))

#%%
# 加权值
y_true = [[0, 1], [0, 0]]
y_pred = [[-18.6, 0.51], [2.94, -12.8]]
y_pred = np.array(y_pred)
y_true = np.array(y_true)

y_prob = sigmoid(y_pred)

# 调用函数
jg = BinaryCrossentropy(from_logits=True)(y_true,y_pred,sample_weight=[0.8,0.2]).numpy()
print(jg)

# 公式计算
r = crossentropy(y_true,y_prob)
print(r)
sample_weight=np.array([0.8,0.2])
print(np.sum(r * sample_weight)/2) # 默认是sum_over_batch_size,是除以总样本数，是4，所以需要再除以2




#%%[markdown]
# categorical crossentropy

#%%
def jcs(y_true,y_pred):
    index_re = []
    for i in range(len(y_true)):
        index_re.append(y_true[i].index(1))
    
    r = []
    for i in range(len(y_true)):
        r.append(-np.log(y_pred[i][index_re[i]]))
    return r
#%%
# Using 'auto'/'sum_over_batch_size' reduction type.

from tensorflow.keras.losses import CategoricalCrossentropy
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

# 调用函数
cce = CategoricalCrossentropy()
jg = cce(y_true, y_pred).numpy()
print(jg)

# 公式计算
r = jcs(y_true,y_pred)
print(np.sum(r)/np.shape(y_true)[0])

#%%
# Calling with 'sample_weight'.
from tensorflow.keras.losses import CategoricalCrossentropy
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

# 调用函数
cce = CategoricalCrossentropy()
jg = cce(y_true, y_pred,sample_weight=tf.constant([0.3, 0.7])).numpy()
print(jg)

# 公式计算
r = jcs(y_true,y_pred)
print(np.sum(np.array(r)*np.array([0.3, 0.7]))/2) # 给损失值加权，然后除以样本数

#%%
# Using 'sum' reduction type.
from tensorflow.keras.losses import CategoricalCrossentropy
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

# 调用函数
cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
jg = cce(y_true, y_pred).numpy()
print(jg)

# 公式计算
r = jcs(y_true,y_pred)
print(np.sum(r)) # 给损失值加权，然后除以样本数

#%%
# Using 'none' reduction type.
from tensorflow.keras.losses import CategoricalCrossentropy
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]

# 调用函数
cce = CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
jg = cce(y_true, y_pred).numpy()
print(jg)

# 公式计算
r = jcs(y_true,y_pred)
print(r) # 给损失值加权，然后除以样本数

#%%
# 累积分布函数cdf
# 激活函数gelu，性能优于relu和elu
import scipy.stats
import numpy as np
import tensorflow as tf

# 调用函数
x = tf.constant([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=tf.float32)
result = tf.nn.gelu(x).numpy()
print(result)

# 根据gelu定义计算
norm_cdf = scipy.stats.norm().cdf(x)
formula_result = x * norm_cdf
print(formula_result)

#%% 
# metrics的accuracy
from tensorflow.keras.metrics import Accuracy
import tensorflow as tf
m = Accuracy()
true = tf.constant([[0,0,1],[0,1,0],[0,1,0]])
predicted = tf.constant([[0,0,1],[0,0,1],[0,1,0]])
m.update_state(true,predicted)
print(m.result().numpy())
print(2.0/3)
print(7.0/9)

#%%
# metrics的BinaryAccuracy
from tensorflow.keras.metrics import BinaryAccuracy
import tensorflow as tf
m = BinaryAccuracy()
m.update_state([[1], [1], [0], [0]], [[0.98], [1], [0], [0.6]])
print(m.result().numpy())

#%%
# 测试fit函数返回的History对象的history属性的内容
# 返回内容包括 训练和验证的 loss 以及 metrics的内容，验证关键字多了"val_"

# history属性的键有dict_keys(['loss', 'bacc', 're', 'val_loss', 'val_bacc', 'val_re'])，
# history属性的数据类型是<class 'dict'>

# 对损失name无用，对指标name使用指标名
import tensorflow as tf
import numpy as np
model = tf.keras.Sequential([tf.keras.Input(shape = (20,)),tf.keras.layers.Dense(1,activation = "sigmoid")])
model.compile(loss = tf.keras.losses.BinaryCrossentropy(name = "bcloss"),optimizer = "adam",metrics = [tf.keras.metrics.BinaryAccuracy(name = "bacc"),tf.keras.metrics.Recall(name = "re")])
print(model.summary())
h = model.fit(np.arange(100).reshape(5,20),np.ones(5),validation_data = (np.arange(100,200).reshape(5,20),np.ones(5)),epochs = 5)
print("history属性的键有{}，history属性的数据类型是{}".format(h.history.keys(),type(h.history)))
print('history属性值为：{}'.format(h.history))
print('模型evaluate输出标量值的标签：{}'.format(model.metrics_names))
# history属性值为：
# {'loss': [70.98738098144531, 69.99639129638672, 69.0053939819336, 68.01438903808594, 67.02339172363281], 'bacc': [0.0, 0.0, 0.0, 0.0, 0.0], 're': [0.0, 0.0, 0.0, 0.0, 0.0], 'val_loss': [195.4595184326172, 192.4685516357422, 189.47752380371094, 186.486572265625, 183.49557495117188], 'val_bacc': [0.0, 0.0, 0.0, 0.0, 0.0], 'val_re': [0.0, 0.0, 0.0, 0.0, 0.0]}

# 模型evaluate输出标量值的标签：['loss', 'bacc', 're']
#%%
# 截取一部分mat数据保存到新mat文件
import scipy.io as sio 
import os
root_path = "F:\\STUDY\\109轴承复合故障原始数据\\外圈+保持架16384"
# root_path = "F:\\STUDY\\109轴承复合故障原始数据\\外圈+保持架+内圈16384"
rpm = ["900rpm","1200rpm","1500rpm"]
suffix = "_1.mat"
name = "oc"
# name = "ioc"

for i in range(3):
    file_path = os.path.join(root_path,rpm[i],rpm[i]+suffix)
    # data_h = sio.loadmat(file_path)['Channel_2_Data']
    # print(data_h.shape) # 列向量

    data_h = sio.loadmat(file_path)['Channel_2_Data'][-1000000:] # 倒着取差不多一分钟的数据
    data_v = sio.loadmat(file_path)['Channel_1_Data'][-1000000:] 
    data_h_name = name + "_" + "H1"
    data_v_name = name + "_" + "V1"
    exec(data_h_name + "=" + "data_h") # 使用字符串作为变量名
    exec(data_v_name + "=" + "data_v")
    sio.savemat(os.path.join(root_path,rpm[i],data_h_name+'.mat'),{data_h_name:data_h}) # 列向量（1000000,1）
    sio.savemat(os.path.join(root_path,rpm[i],data_v_name+".mat"),{data_v_name:data_v})

#%%
import tensorflow as tf

class MyModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)

model = MyModel()
model.compile(loss= 'categorical_crossentropy',metrics=["mse",'accuracy'],optimizer = "adam")
inputs = tf.constant([[1,2,3,4],[1,1,3,4]])
outputs = tf.constant([[0,0,0,0,1],[1,0,0,0,0]])
model.fit(inputs,outputs,epochs = 1)
print("distribute_strategy is {}".format(model.distribute_strategy))
print("layers 属性{}".format(model.layers))
print("model的metrics属性{}".format(model.metrics))
# distribute_strategy is <tensorflow.python.distribute.distribute_lib._DefaultDistributionStrategy object at 0x0000014BC938E7C8>
# layers 属性[<tensorflow.python.keras.layers.core.Dense object at 0x0000014BC8926488>, <tensorflow.python.keras.layers.core.Dense object at 0x0000014BC8B6CE08>]
# model的metrics属性[<tensorflow.python.keras.metrics.Mean object at 0x0000014BC94F7D08>, <tensorflow.python.keras.metrics.MeanMetricWrapper object at 0x0000014BCA319F08>, <tensorflow.python.keras.metrics.MeanMetricWrapper object at 0x0000014BCB37BC08>]

#%%
import tensorflow as tf
class SimpleDense(tf.keras.layers.Layer):

  def __init__(self, units=32):
      super(SimpleDense, self).__init__()
      self.units = units

  def build(self, input_shape):  # Create the state of the layer (weights)
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
        initial_value=w_init(shape=(input_shape[-1], self.units),
                             dtype='float32'),
        trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
        initial_value=b_init(shape=(self.units,), dtype='float32'),
        trainable=True)

  def call(self, inputs):  # Defines the computation from inputs to outputs
      return tf.matmul(inputs, self.w) + self.b

# Instantiates the layer.
linear_layer = SimpleDense(4)

# This will also call `build(input_shape)` and create the weights.
y = linear_layer(tf.ones((2, 2)))
assert len(linear_layer.weights) == 2

# These weights are trainable, so they're listed in `trainable_weights`:
assert len(linear_layer.trainable_weights) == 2

#%%
import tensorflow as tf
import os

checkpoint_directory = "/tmp/training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
print(checkpoint_prefix)
