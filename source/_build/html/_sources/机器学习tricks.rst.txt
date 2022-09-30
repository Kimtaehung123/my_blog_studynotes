5.9 机器学习tricks
------------------

1. 约束项的使用constraints

    constraints模块的函数允许在优化期间对网络参数设置约束（例如非负性）。
    其实就是优化问题对可调参数设置了取值范围约束。
    常见的约束项包括：非负约束、范数大小约束等。

2. 正则化器的使用regularizer

    正则化器允许在优化过程中对 **层的参数** 或 **层的激活情况** 进行惩罚penalty。
    网络优化的损失函数也包括这些惩罚项。
    即基本的损失函数是 **目标标签和模型预测标签之间的差异衡量**，正则化器并不是描述这种差异的，但是加入到了损失函数中，影响了优化过程，提高了模型的泛化能力。
    常见的正则化器就是L1正则化器、L2正则化器、L1和L2正则化器
    You can access a layer's regularization penalties by calling layer.losses after calling the layer on inputs.

    



    