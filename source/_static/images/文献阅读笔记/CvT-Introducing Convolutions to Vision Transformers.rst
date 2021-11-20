@microsoft.com

**Abstract**:

- 在性能和效率上改进Vision Transformer(ViT)
- 结合ViT和卷积，结合两者的优点，产生最佳的模型
- 主要通过两步：step1-包含 :code:`卷积符号嵌入convolutional token embedding` 的Transformer层级结构；
  step2-利用卷积映射的卷积Transformer块。
- 卷积网络的特点（平移、伸缩、扭曲不变性）；Transformer的优点（动态注意力、全局上下文、更好的泛化能力）
- 模型、参数量、浮点计算次数等方面验证所提方法的有效性；此外，采用预训练-微调方法性能也是好的；最后，在ViT中的位置编码可以在所提模型中移除，简化了对高分辨率视觉图像任务的设计。

**Introduction**：

- ViT是第一个仅依赖Transformer架构，在大尺度数据集上获得了较强的图像分类性能；
- ViT把图像分成不重叠的块（类似NLP中的符号token），再叠加特殊的位置编码信息表示粗粒度的空间信息，最后输入到重复的标准Transformer层中，对分类任务建立全局联系建模；
- 在小样本数据上，ViT的性能仍低于相似尺寸的卷积网络。

**Related Work**:

- Vision Transformer
- Introducing Self-attentiions to CNNs
- Introducing Convolutions to Transformers

**Convolutional vision Transformer**:

- Convolutional Token Embedding 
- Convolutional Projection for Attention 
    - Implementation Details
    - Efficiency Considerations
- Methodological Discussions
    - Removing Positional Embeddings
    - Relations to Concurrent Work

**Experiments**:

- Setup
    - Model Variants
    - Training
    - Fine-tuning
- Comparison to state of the art 
- Downstream task transfer 
- Ablation Study
    - Removing Position Embedding
    - Convolutional Token Embedding
    - Convolution Projection

**Conclusion**:

