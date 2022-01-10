5.4.1 vanilla transformer 
--------------------------

5.4.1.1 参考
--------------

`vanilla transformer <https://blog.csdn.net/sinat_37574187/article/details/119890682>`_

`论文笔记-Vanilla Transformer：Character-Level Language Modeling with Deeper Self-Attention <https://blog.csdn.net/u010366748/article/details/114301942>`_

5.4.1.2 详情
---------------

* **Vanilla Transformer的三个缺点：**
    * 上下文长度受限：字符之间的最大依赖距离受输入长度的限制，模型看不到出现在几个句子之前的单词。  
    * 上下文碎片：对于长度超过512个字符的文本，都是从头开始单独训练的。段与段之间没有上下文依赖性，会让训练效率低下，也会影响模型的性能。
    * 推理速度慢：在测试阶段，每次预测下一个单词，都需要重新构建一遍上下文，并从头开始计算，这样的计算速度非常慢。

* **模型架构：**：
    * 模型架构上基本就是堆叠的64层transformer layers，每个layer有2个head。
    * 一个transformer layer就是指包含multihead self-attention sub-layer + 
    * 2层全连接sub-layers的 feed-forward network。
    * 每层Transformer layer的hidden_size=512， 
    * feed-forward network的内层FFC的dim是2048。
    * 模型处理的序列长度为512。


5.4.1.3 tips 
-----------------

* 字符级别（Character-Level ） 和 词级别【NLP领域术语】
* vanilla在计算机领域指的就是最原始的意思。