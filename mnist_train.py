# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference


# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 30000

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/"
MODEL_NAME = "model.ckpt"


def train(mnist):
    # 定义输入输出placeholder（占位符），输入为一个四维矩阵
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,  # 第一维表示一个batch中样例的个数
        mnist_inference.IMAGE_SIZE,  # 第二维和第三维表示图片的尺寸
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],  # 第四维表示图片的深度
                       name='x-input')
    # None表示第一维是任意数量，可以方便使用不同的batch大小
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    # 直接使用mnist_inference.py中定义的前向传播过程
    y = mnist_inference.inference(x, True)
    global_step = tf.Variable(0, trainable=False)   # 迭代iteration的计数器

    # 定义损失函数、学习率以及训练过程
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数，这里使用Tensorflow中提供的
    # sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类问题只有一个正确答案时，
    # 可以使用这个函数来加速交叉熵的计算。这个函数的第一个参数是神经网络不包括softmax的前向传播结果
    # 第二个是训练数据的正确答案。因为标准答案是一个长度为10的一维数组，而该函数需要提供的是
    # 一个正确答案的数字，所以需要使用tf.argmax(y_,1)函数来得到正确答案的编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算在当前batch中所有样例的交叉熵平均值。
    loss = cross_entropy_mean
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY, staircase=True)     # 学习率下降，每训练550轮，学习率乘以0.99
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)   # 小批量梯度下降

    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:  # 创建一个会话
        tf.global_variables_initializer().run()     # 初始化模型的参数
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)     # 通过next_batch()就可以一个一个batch的拿数据
            # 类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))
            # 占位符并没有初始值，它只会分配必要的内存。在会话中，占位符可以使用feed_dict馈送数据。
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})
            # 每100轮保存一次模型
            if i % 100 == 0:
                # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况
                # 在验证数据集上的正确率信息会有一个单独的程序来生成
                print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数
                # 比如“model.ckpt-1000”表示训练1000轮后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("dataset/", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
