5.4.2 a_survey_of_transformers
-------------------------------------

**主要内容：**
   
   a. vanilla Transformer &  X-formers
   b. X-formers from three perspectives: 
       * architectural modification
       * pre-training
       * applications
   c. potential directions

**关键词：**

   a. Transformer
   b. Self-Attention
   c. Pre-trained Models
   d. Deep Learning

1. **Introduction:**

   * *fields* : NLP, CV, speech processing,Transformer-based pre-trained models (PTMs)
   * *origin* : 机器翻译的序列到序列模型
   * *X-formers improvements* ：These X-formers improve the vanilla Transformer from different perspectives:
        
        * model efficience : 
                      
              long sequences, computation and memory complexity of self-attention
        
        * model generalization: 
          
              makes few assumptions on the structural bias of input data, 
              it is hard to train on small-scale data.
        
        * model adaptation:

              adapt the Transformer to specific downstream tasks and applications.

    * *organization of the article* :

        * Sec. 2 : the architecture and the key components of Transformer. 
        * Sec. 3 : clarifies the categorization of Transformer variants. 
        * Sec. 4∼5 : review the module-level modifications, including attention module, position encoding, layer normalization and feed-forward layer. 
        * Sec. 6 : reviews the **architecture-level variants**. 
        * Sec. 7 : introduces some of the representative **Transformer-based PTMs** (pre-trained models). 
        * Sec. 8 : introduces the **application of Transformer** to various different fields. 
        * Sec. 9 : discusses some aspects of Transformer that researchers might find intriguing and summarizes the paper.

2. **background:**

    * *vanilla transformer* :

        * multi-head self-attention module(MHA)(self-attention modules)(masked self-attention modules)(cross-attention modules)
        * a position-wise feed-forward network(FFN)
        * residual connection(deeper model) & layer normalization module 
        * position encoding
    * 