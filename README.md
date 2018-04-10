# MNIST_Tensorflow

## MNIST_Tensorflow工程目录

* -dataset                           //存放数据集的文件夹<br>
* -model                             //存放模型的文件夹<br>
* -mnist_inference.py     //定义了前向传播的过程以及神经网络中的参数<br>
* -mnist_train.py               //定义了神经网络的训练过程<br>
* -mnist_eval.py                //定义了测试过程 


## MNIST 数据集
| 文件 | 内容 | 
| - | :-: |
| train-images-idx3-ubyte.gz | 训练集图片：55000张训练图片，5000 张验证图片 |
| train-labels-idx1-ubyte.gz | 训练集图片对应的数字标签 | 
| t10k-images-idx3-ubyte.gz | 测试集图片：10000张图片 |
| t10k-labels-idx1-ubyte.gz | 测试集图片对应的数字标签 |


## Net网络结构：
![Alt text](https://upload-images.jianshu.io/upload_images/11573712-ee2cfafeb0c48db6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/700)
* INPUT: [28x28x1]                weights: 0<br>
* CONV5-32: [28x28x32]     weights: (5 * 5 * 1 + 1) * 32<br>
* POOL2: [14x14x32]              weights: 0<br>
* CONV5-64: [14x14x64]      weights: (5 * 5 * 32 + 1) * 64<br>
* POOL2: [7x7x64]                  weights: 0<br>
* FC: [1x1x512]                        weights: (7 * 7 * 64 + 1) * 512<br>
* FC: [1x1x10]                          weights: (1 * 1 * 512 + 1) * 10<br>




  
