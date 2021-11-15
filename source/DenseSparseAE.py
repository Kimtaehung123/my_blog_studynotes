# -*- coding: utf-8 -*-
# %% [markdown]
# #### 数据和标签（中介轴承频谱数据）
# 每个.npz数据文件中只有一个关键词，为'data'，.npz数据对.mat中数据做了转置
# 全为频谱数据
# <**每个文件中的数据类型**>
# 正常（1-200）、外圈（201-400）、内圈（401-600）、滚动体（601-800）
# 每个文件中都有800条数据，每条数据长度为8192

# %%
import numpy as np
from tensorflow.keras.utils import to_categorical

data_path = "/home/u2020200708/DATASETS/中介轴承频谱数据（崔）/dataNOIR300_600pinpu.npz"
data = np.load(data_path)['data']
# print(data.shape)
# print(np.max(data),np.min(data)) # 检查结果，输入在[0-1]之间，使用relu的激活函数可以编码解码输入数据
train_data = np.concatenate((data[0:20][:],data[200:220][:],data[400:420][:],data[600:620]))
val_data = np.concatenate((data[60:80][:],data[260:280][:],data[460:480][:],data[660:680])) 
test_data = np.concatenate((data[100:200][:],data[300:400][:],data[500:600][:],data[700:800])) 

train_label = to_categorical([i for i in range(4) for j in range(20)],num_classes=4)
val_label = to_categorical([i for i in range(4) for j in range(20)],num_classes=4)
test_label = to_categorical([i for i in range(4) for j in range(100)],num_classes=4)

# CWRU
# data_path = "/home/u2020200708/SparseAE/DenseSparseAE_L1/EXPERIMENTS/CWRU/time_domain_test1/data"
# train_data = np.load(data_path + "/trainset.npz")["train_data"]
# val_data = np.load(data_path + "/validate.npz")["validate_data"]
# test_data = np.load(data_path + "/testset.npz")["test_data"]

# train_label = np.load(data_path + "/trainset.npz")["train_label"]
# val_label = np.load(data_path + "/validate.npz")["validate_label"]
# test_label = np.load(data_path + "/testset.npz")["test_label"]

# %% [markdown]
# #### 对隐层激活值使用L1正则化 进行稀疏控制
# 使用一维卷积的稀疏自编码器
# 损失函数使用mse，激活函数使用RELU，优化器使用Adam来训练自编码器
# 之后使用交叉熵损失，加上分类层，进行分类任务
# 逐层贪婪训练

# %%
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D,Input
import matplotlib.pyplot as plt
import numpy as np

def denseSparseAE_train(
    seq_length,
    hidden_nums,
    train_data,
    batch_size,
    epochs,
    val_data,
    save_root
):
    '''
    Construct and train *Stacked Sparse AutoEncoder*.

    ** Params:**
    -------------------

    - seq_length:

        int,length of the 1-d samples.

    - hidden_nums:

        list,number of neurons for each hidden dense layer.

    '''
    '''
    -------------------------
    存储容器
    --------------------------
    '''
    inputs_nums = [seq_length] + hidden_nums[:-1] # 每个自编码器的输入维度
    print(inputs_nums)
    subAE_history = []
    subAE_model = []

    '''
    ----------------------------------
    循环构建训练单个稀疏自编码器
    -----------------------------------
    '''
    for i in range(len(hidden_nums)):
        x = Input(shape = (inputs_nums[i],)) # shape is a tuple,not including the batch size.
        y = Dense(hidden_nums[i],activation='relu',activity_regularizer="l2")(x) # 隐层加入激活值L1正则化器
        y = Dense(inputs_nums[i],activation='relu')(y) # relu结果只有正数，但是输入是有正有负的，所以需要把输入在训练前先归一化到0-1之间

        model = Model(x,y,name = "DenseSparseAE"+str(i)) # 得到一个全连接的稀疏自编码器

        model.compile(optimizer="adam",loss='mse') # 这种输出拟合输入没有“准确率”一说，分类任务有

        H = model.fit(train_data,train_data,batch_size= batch_size,epochs=epochs,validation_data=(val_data,val_data))

        subAE_history.append(H) # 记录子编码器 训练的历史记录
        subAE_model.append(model) # 记录子编码器的 模型

        if i != len(hidden_nums) - 1:
            sub_coder = Model(model.input,model.layers[1].output) # 得到子编码器的“隐层输出模型”，预测输出作为 下一个自编码器的输入来进行逐层贪婪训练
            train_data = sub_coder.predict(train_data) # 下一个自编码器的训练数据
            val_data = sub_coder.predict(val_data) # 下一个自编码器的验证数据
    '''
    ----------------------------------
    多层自编码器和解码器的组合、保存
    ----------------------------------
    '''
    # 把所有子编码器 的编码层 组合 —— 最终完整的编码器
    x = Input(shape=(seq_length,))
    y = x
    for i in range(len(hidden_nums)):
        y = subAE_model[i].layers[1](y) # 把层提取出来，使用functional API组合为模型
    coder = Model(x,y)

    coder.save(save_root + "/coder.h5") # 保存编码器

    # 把所有子解码器 的解码层 组合 —— 最终完整的解码器
    x = Input(shape=(hidden_nums[-1],)) # 解码器输入长度 为 最后一个隐层神经元数
    y = x
    for i in range(len(hidden_nums)):
        y = subAE_model[-1-i].layers[2](y) 
    decoder = Model(x,y)

    decoder.save(save_root + "/decoder.h5") # 保存解码器
    '''
    ------------------------------------------
    对每个子编码器的训练损失曲线画图
    ------------------------------------------    
    '''
    for i in range(len(hidden_nums)):
        plt.figure()
        plt.title("Activation L1-regularized Single AE _" + str(i+1))
        plt.xlabel("Epochs")
        plt.ylabel("MSE-Loss")
        plt.plot(np.arange(1,epochs+1,1),subAE_history[i].history['loss'],label = "train_loss_"+str(round(subAE_history[i].history['loss'][-1],4)))
        plt.plot(np.arange(1,epochs+1,1),subAE_history[i].history['val_loss'],label = "val_loss_"+str(round(subAE_history[i].history['loss'][-1],4)))
        plt.legend()
        plt.savefig(save_root + "/training_curve_subAE_" + str(i+1) + ".png",dpi = 500,bbox_inches = "tight")


# %% [markdown]
# #### 固定预训练编码器，使用微调新增分类器层

import tensorflow as tf
from tensorflow.keras.models import Model,load_model 
from tensorflow.keras.layers import Input,Dense,Dropout 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

def classify_model_training(
    train_data,
    train_label,
    val_data,
    val_label,
    test_data, 
    test_label,
    save_root,
    seq_length,
    classifier_dense_nums,
    class_num,
    batch_size,
    epochs,
    class_names
):
    '''
    ----------
    Params:
    ----------
    - save_root:

        str,root path to store the results.

    - seq_length:

        int,the length of one-dim data,e.g.8192

    - classifier_dense_nums:

        list of neuron numbers in dense layers of classifier(except the last 'softmax' layer),e.g.[20,10]

    - class_num:

        int,the number of classes,e.g.4

    - class_names:

        list of strings of each class name,e.g.["N","IR","OR","R"]

    -------------
    Returns:
    ---------------

    summary of the classify model.
    '''

    '''
    -------------------------
    构建微调分类器模型
    -------------------------
    '''
    # 导入模型
    pretrained_coder_path = save_root + "/coder.h5" 
    coder = load_model(pretrained_coder_path)

    # 构建微调模型
    inputs = Input(shape=(seq_length,))
    x = coder(inputs,training = False) # 冻结编码器的参数，设置为不可训练
    for i in range(len(classifier_dense_nums)):
        x = Dense(classifier_dense_nums[i],activation='relu')(x)
        x = Dropout(0.5)(x)
    x = Dense(class_num,activation='softmax')(x)
    classify_model = Model(inputs,x)

    '''
    ----------------------------------
    编译、训练、保存分类模型
    ---------------------------------
    '''
    # 编译，训练，保存模型
    classify_model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

    H = classify_model.fit(train_data,train_label,batch_size=batch_size,epochs=epochs,validation_data=(val_data,val_label))

    classify_model.save(save_root + "/classify_model.h5")

    '''
    -----------------------------------
    画分类模型训练过程损失/准确率曲线图
    -----------------------------------
    '''
    plt.figure()
    plt.title("classify model training loss/accuracy curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.plot(np.arange(1,epochs+1,1),H.history["loss"],label = "train_loss_"+str(round(H.history["loss"][-1],4)))
    plt.plot(np.arange(1,epochs+1,1),H.history["val_loss"],label = "val_loss_"+str(round(H.history["val_loss"][-1],4)))
    plt.plot(np.arange(1,epochs+1,1),H.history["accuracy"],label = "train_accuracy_"+str(round(H.history["accuracy"][-1],4)))
    plt.plot(np.arange(1,epochs+1,1),H.history["val_accuracy"],label = "val_accuracy_"+str(round(H.history["val_accuracy"][-1],4)))
    plt.legend()
    plt.savefig(save_root + "/classify model training curve.png")

    '''
    -------------------------------------------------
    使用测试集验证分类模型的效果，得到多个分类性能指标
    -------------------------------------------------
    '''
    predictions = classify_model.predict(test_data)
    print(classification_report(test_label.argmax(axis = 1),predictions.argmax(axis = 1),target_names = class_names))

    return classify_model.summary()
# %% [markdown]
# #### 代入数据，预训练自编码器，微调整个分类模型

# %%
denseSparseAE_train(
    8192,
    [12000],
    train_data,
    100,
    100,
    val_data,
    "/home/u2020200708/SparseAE/DenseSparseAE_L1L2/EXPERIMENTS/中介轴承频谱数据实验/dataNOIR300_600pinpu/result_2")

classify_model_summary = classify_model_training(
    train_data,
    train_label,
    val_data,
    val_label,
    test_data,
    test_label,
    "/home/u2020200708/SparseAE/DenseSparseAE_L1L2/EXPERIMENTS/中介轴承频谱数据实验/dataNOIR300_600pinpu/result_2",
    8192,
    [20],
    4,
    100,
    600,
    ['normal','outer','inner','rolling']
    )

print(classify_model_summary)