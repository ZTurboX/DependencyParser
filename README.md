## implement of "A Fast and Accurate Dependency Parser using Neural Networks "

### 0. 环境
- python3.6

### 1. run code

- prepare_data.py: 此文件为数据预处理过程

- train the model : python main.py --mode train

- 训练完成后，将模型拷贝至./data/model文件夹，并命名为model.pt

- test the model: python main.py --mode test

-----------------------------------------------------------------------------------------------------------------
### 2. 实验记录
- #### 2019.05.03

在word embedding后加入dropout

|      | test |
| :--: | :--: |
| UAS  | 81.3 |

-------------------------------------------------------------------------------------------------------------------------------------------------

- #### 2019.04.30

修改激活函数、超参数

加入正则化

activation function : relu

optimization : Adagrad(lr=0.01,weight_decay=1e-8)

|      | test |
| :--: | :--: |
| UAS  | 79.1 |

-------------------------------------------------------------------------------------------------------------------------------------------------

- #### 2019.4.29

修改特征抽取函数

activation function : cube

batch size ：2048

|      | test |
| :--: | :--: |
| UAS  | 77.6 |

-------------------------------------------------------------------------------------------------------------------------------------------------

- #### 2019.4.25

activation function：relu

optimization : mini-batched Adam

dropout : 0.5

hidden size : 200

batch size ； 1024

word embedding size : 100

pretrained word embedding : glove

|      | test |
| :--: | :--: |
| UAS  | 74.1 |

---------------------------------------------------------------------------------------------------------------------------------------------


