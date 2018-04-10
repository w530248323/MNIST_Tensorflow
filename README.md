# MNIST_Tensorflow

## MNIST_Tensorflow工程目录

* -dataset                           //存放数据集的文件夹<br>
* -model                             //存放模型的文件夹<br>
* -mnist_inference.py     //定义了前向传播的过程以及神经网络中的参数<br>
* -mnist_train.py               //定义了神经网络的训练过程<br>
* -mnist_eval.py               //定义了测试过程<br>  

## Net网络结构：
* INPUT: [28x28x1]                  weights: 0<br>
* CONV5-32: [28x28x32]     weights: (5*5*1+1)*32<br>
* POOL2: [14x14x32]              weights: 0<br>
* CONV5-64: [14x14x64]      weights: (5*5*32+1)*64<br>
* POOL2: [7x7x64]                  weights: 0<br>
* FC: [1x1x512]                        weights: (7*7*64+1)*512<br>
* FC: [1x1x10]                          weights: (1*1*512+1)*10<br>
  
  
