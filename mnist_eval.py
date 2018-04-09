# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 加载mnist_inference.py 和 mnist_train.py中定义的常量和函数
import mnist_inference
import mnist_train

# 每10秒加载一次最新的模型， 并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default():
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [
            mnist.validation.num_examples,  # 第一维表示样例的个数
            mnist_inference.IMAGE_SIZE,     # 第二维和第三维表示图片的尺寸
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],  # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

        validate_feed = {x: np.reshape(mnist.validation.images, (mnist.validation.num_examples,
                                                                 mnist_inference.IMAGE_SIZE,
                                                                 mnist_inference.IMAGE_SIZE,
                                                                 mnist_inference.NUM_CHANNELS)),
                         y_: mnist.validation.labels}
        # 直接通过调用封装好的函数来计算前向传播的结果。
        y = mnist_inference.inference(x, False)
        # 如果需要对未知的样例进行分类，使用tf.argmax(y, 1)能给出某个tensor对象在某一维上的其数据最大值所在的索引值。
        # 由于标签向量是由0,1组成，因此最大值1所在的索引位置就是类别标签，
        # 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 将布尔值转换为浮点数来代表对、错，然后取平均值。例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()

        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)     # 加载模型参数
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('-')[-1]     # ？？？？？
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %f" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets("dataset/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
