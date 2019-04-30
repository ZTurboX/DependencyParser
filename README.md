implement of "A Fast and Accurate Dependency Parser using Neural Networks "

-----------------------------------------------------------------------------------------------------------------

2019.4.25

activation function：relu

optimization : mini-batched Adam

dropout : 0.5

hidden size : 200

batch size ； 1024

word embedding size : 100

pretrained word embedding : glove

|      | dev  | test |
| :--: | :--: | :--: |
| UAS  | 73.6 | 74.1 |

---------------------------------------------------------------------------------------------------------------------------------------------

2019.4.29

activation function : cube

batch size ：2048

|      | dev  | test |
| ---- | ---- | ---- |
| UAS  | 77.3 | 77.6 |

---------------------------------------------------------------------------------------------------------------------------------------------

2019.04.30

activation function : relu

optimization : Adagrad(lr=0.01,weight_decay=1e-8)

|      | dev  | test |
| ---- | ---- | ---- |
| UAS  | 78.8 | 79.1 |