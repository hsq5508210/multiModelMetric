#-*-coding:utf-8-*-

import tensorflow as tf
import math
from tensorflow.python.platform import flags
from utils import conv_block, distance, mse, get_dist_category, get_acc, intra_dist, inter_dist
FLAGS = flags.FLAGS



class Model:
    def __init__(self, sess, output_dim=10):
        self.input_dim = FLAGS.input_dim
        self.output_dim = output_dim
        self.dim_hidden = FLAGS.filter_num
        self.neighbor_k = FLAGS.k_neighbor
        self.lr = FLAGS.lr
        self.sess = sess
        if FLAGS.data_source == 'PACS':
            self.channels = 3
        if FLAGS.backbone == 'Conv':
            self.construct_weights = self.construct_conv
            self.forward = self.forward_conv
        else:
            self.construct_weights = self.construct_res
            self.forward = self.forward_res
    def decay(self):
        global_step = FLAGS.episode_tr * FLAGS.iteration
        decay_steps = FLAGS.iteration
        decay_rate = FLAGS.decay_rate
        self.lr = tf.train.exponential_decay(self.lr, global_step=global_step, decay_steps=decay_steps,
                                              decay_rate=decay_rate)
    def get_loss(self, inp, weights, resuse=True):
        with tf.name_scope("compute_loss"):
            support_x, support_y, query_x, query_y = inp
            output_s = self.forward(self.support_x, weights, reuse=resuse)
            output_q = self.forward(self.query_x, weights, reuse=resuse)
            predict = self.category_choose(output_q, output_s,  support_y)
            accurcy = get_acc(predict, query_y)
            intradist =
            task_losses = \

        return task_losses, accurcy


    def construct_model(self, input_tensor=None):
        if input_tensor is None:
            with tf.name_scope("input"):
                self.support_x = tf.placeholder(tf.float32, name='support_x')
                self.support_y = tf.placeholder(tf.float32, name='support_y')
                self.query_x = tf.placeholder(tf.float32, name='query_x')
                self.query_y = tf.placeholder(tf.float32, name='query_y')
        else:
            self.support_x = input_tensor['support_set'][0]
            self.support_y = input_tensor['support_set'][1]
            self.query_x = input_tensor['query_set'][0]
            self.query_y = input_tensor['query_set'][1]
        if 'weights' in dir(self):
            weights = self.weights
        else:
            self.weights = weights = self.construct_weights()
        accurcy = []
        losses = []

    def trainop(self):
        losses, acc = self.get_loss((self.support_x, self.support_y, self.query_x, self.query_y), self.weights)
        if FLAGS.train:
            with tf.name_scope("loss"):
                self.loss = loss = tf.reduce_sum(losses) / tf.to_float(FLAGS.query_num * FLAGS.model * FLAGS.way_num)
            with tf.name_scope("compute_grad"):
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.gvs = gvs = optimizer.compute_gradients(loss+0.00001)
                with tf.name_scope("clip_grad"):
                    if FLAGS.data_source == 'PACS':
                        gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs]
                self.train = optimizer.apply_gradients(gvs)
            self.acc = acc
        return self.train, [self.loss, acc]





    def category_choose(self, output_q, output_s, label_s):
        with tf.name_scope('category_choose'):

            output_s = self.vectorlize(output_s)
            output_q = self.vectorlize(output_q)
            with tf.name_scope('category'):
                softmax = tf.map_fn(fn=lambda q:get_dist_category(q, output_s, label_s), elems=output_q)
        return softmax



    def vectorlize(self, x):
        with tf.name_scope('vectorlize'):
            size = math.floor(FLAGS.image_size/(2**4))
            x = tf.reshape(tensor=x, shape=[-1, size*size*FLAGS.filter_num])
        return x

    def construct_conv(self):
        weights = {}
        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype,seed=0)
        k = 3
        with tf.name_scope('weights'):
            weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
            weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
            weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
            weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        return weights


    def forward_conv(self, inp, weights, reuse=True):
        scope = ''
        with tf.name_scope("forward_conv"):
            hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse)
            hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse)
            hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse)
            hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse)
        return hidden4


    def construct_res(self):
        pass

    def forward_res(self, inp, weights, reuse=False):


        pass



