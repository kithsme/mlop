import tensorflow as tf
import numpy as np
import random
import params
import utils
import genImg
import trackAnalysis as TA
import orderAnalysis as OA 
import pandas as pd 

prm = params.params()

def get_samples():

    def unpack_tuples(tuples):
        xs, ys = [], [] 
        for x,y in tuples:
            xs.append(x)
            if y == 1:
                y = [1., 0.]
            else:
                y = [0.,1.]
            ys.append(y)
        
        xs = np.asarray(xs)
        ys = np.asarray(ys)

        return xs, ys

    xys = genImg.prepareLearningXYsFromFiles('C:/Users/pos/Desktop/png/pos', 'C:/Users/pos/Desktop/png/neg')
    random.shuffle(xys)

    train_xy = xys[:int(len(xys)*prm.training_set_size)]
    test_xy = xys[int(len(xys)*prm.test_set_size):]

    train_x, train_y = unpack_tuples(train_xy)
    test_x, test_y = unpack_tuples(test_xy)

    return train_x, train_y, test_x, test_y


class ConvNet:

    def __init__(self, session, name):

        self.session = session
        self.net_name = name
        self.build_network()

    def build_network(self):

        def conv2d(x,W,s):
            return tf.nn.conv2d(x,W, strides=[1,s,s,1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        with tf.variable_scope(self.net_name):

            self._x = tf.placeholder(tf.float32, shape=[None, prm.step, prm.step, 3], name='input_x')
            self._y = tf.placeholder(tf.float32, shape=[None, 2], name='label')

            W_conv1 = tf.get_variable('W1', shape=[prm.conv1_filter_size, prm.conv1_filter_size, 3, prm.conv1_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            b_conv1 = tf.get_variable('b1', shape=[prm.conv1_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            h_conv1 = tf.nn.relu(conv2d(self._x, W_conv1, prm.conv1_stride) + b_conv1, name='h1')
            
            W_conv2 = tf.get_variable('W2', shape=[prm.conv2_filter_size, prm.conv2_filter_size, prm.conv1_output_size , prm.conv2_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            b_conv2 = tf.get_variable('b2', shape=[prm.conv2_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, prm.conv2_stride) + b_conv2, name='h2')
            
            W_conv3 = tf.get_variable('W3', shape=[prm.conv3_filter_size, prm.conv3_filter_size, prm.conv2_output_size , prm.conv3_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            b_conv3 = tf.get_variable('b3', shape=[prm.conv3_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, prm.conv3_stride) + b_conv3, name='h3')
            
            W_conv4 = tf.get_variable('W4', shape=[prm.conv4_filter_size, prm.conv4_filter_size, prm.conv3_output_size , prm.conv4_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            b_conv4 = tf.get_variable('b4', shape=[prm.conv4_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, prm.conv4_stride) + b_conv4, name='h4')
            
            W_conv5 = tf.get_variable('W5', shape=[prm.conv5_filter_size, prm.conv5_filter_size, prm.conv4_output_size , prm.conv5_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
           
            b_conv5 = tf.get_variable('b5', shape=[prm.conv5_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
           
            h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, prm.conv5_stride) + b_conv5, name='h5')
            
            print(h_conv5.shape)
            h_conv5_flat = tf.reshape(h_conv5, [-1, int(h_conv5.shape[1])*int(h_conv5.shape[2])*prm.conv5_output_size])

            print(h_conv5_flat.shape)
            W_fc1 = tf.get_variable('Wfc1', shape=[int(h_conv5.shape[1])*int(h_conv5.shape[2])*prm.conv5_output_size, prm.fc1_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())
            
            b_fc1 = tf.get_variable('bfc1', shape=[prm.fc1_output_size],
                                      initializer=tf.contrib.layers.xavier_initializer())


            h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1)+b_fc1, name='hfc1')

            self.keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

            W_fc2 = tf.get_variable('Wfc2', shape=[prm.fc1_output_size, 2],
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_fc2 = tf.get_variable('bfc2', shape=[2],
                                      initializer=tf.contrib.layers.xavier_initializer())

            self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self._y, logits=self.y_conv)
            )

            self.train = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def predict(self, x):
        return self.session.run(self.y_conv, feed_dict={self._x:x, self.keep_prob:1.0})

    # def train(self, x, y, k):
    #     return self.session.run(self.train, feed_dict={self._x:x, self._y: y, self.keep_prob:k})

tr_x, tr_y, te_x, te_y = get_samples()

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    conv = ConvNet(sess, 'mainConv')
    tf.global_variables_initializer().run()

    for i in range(10):
        conv.train.run(feed_dict={conv._x: tr_x[100*i:100*i+100], conv._y: tr_y[100*i:100*i+100], conv.keep_prob:1.0})
    
    print(conv.predict(te_x[0:10]))
    print(te_y[0:10])