#-*-coding:utf-8-*-

import tensorflow as tf
import math
from tensorflow.python.platform import flags
from utils import conv_block, get_acc, loss_eps, category_choose, category_loss, vectorlize, distance
FLAGS = flags.FLAGS



class Model:
    def __init__(self, output_dim=10):
        self.input_dim = FLAGS.input_dim
        self.output_dim = output_dim
        self.dim_hidden = FLAGS.filter_num
        self.neighbor_k = FLAGS.k_neighbor
        self.lr = tf.convert_to_tensor(FLAGS.lr, dtype=tf.float32)
        self.w = FLAGS.loss_weight
        # self.sess = sess
        # if FLAGS.data_source == 'PACS':
        self.channels = 3
        if FLAGS.backbone == 'Conv':
            self.construct_weights = self.construct_conv
            self.forward = self.forward_conv
        else:
            self.construct_weights = self.construct_res
            self.forward = self.forward_res
        if FLAGS.loss_function == 'mse':
            self.loss_function = category_loss
    def decay(self):
        global_step = FLAGS.episode_tr * FLAGS.iteration
        decay_steps = FLAGS.iteration/100
        decay_rate = FLAGS.decay_rate
        self.lr = tf.train.exponential_decay(self.lr, global_step=global_step, decay_steps=decay_steps,
                                              decay_rate=decay_rate)
    def get_loss(self, inp, resuse=True):
        with tf.name_scope("compute_loss"):
            weights = self.weights
            support_x, support_y, query_x, query_y, support_m, query_m = inp
            output_s = self.forward(support_x, weights, reuse=resuse)
            output_q = self.forward(query_x, weights, reuse=resuse)
            self.predict = predict = category_choose(output_q, output_s,  support_y)
            accurcy = get_acc(predict, query_y)
            # task_losses = tf.map_fn(fn=lambda qxy: self.loss_function(qxy, output_s, support_y, 0.4),
            #                         elems=(output_q, query_y), dtype=tf.float32,
            #                         parallel_iterations=FLAGS.model * FLAGS.way_num * FLAGS.query_num)
            # task_losses = self.loss_function((output_q, query_y), output_s, support_y)
            task_losses = self.loss_function(predict, query_y)
            if FLAGS.eps_loss and FLAGS.category_loss:
                self.losses_eps = losses_eps = loss_eps(output_s, output_q, support_m, query_m, support_y,
                                      query_y, FLAGS.margin)
                losses = (1 - self.w) * task_losses + self.w * losses_eps
                return losses, accurcy
            elif FLAGS.category_loss:
                return task_losses, accurcy
            elif FLAGS.eps_loss:
                self.losses_eps = losses_eps = loss_eps(output_s, output_q, support_m, query_m, support_y,
                                                        query_y, FLAGS.margin)
                return losses_eps, accurcy
    def debuf_nan(self, output_q, output_s):
        output_s = vectorlize(output_s)
        output_q = vectorlize(output_q)
        # dist = distance(output_q, output_s)
        # res = tf.exp(-tf.matmul(dist, self.support_y))
        # sum = tf.diag(1/tf.reduce_sum(res, axis=1))
        # softmax = tf.matmul(sum, res)

        # predict = category_choose(output_q, output_s, self.support_y)
        # task_losses = self.loss_function(predict, self.query_y)
        # losses_eps = loss_eps(output_s, output_q, self.support_m, self.query_m, self.support_y,
        #                       self.query_y, FLAGS.margin)

        losses, acc = self.get_loss((self.support_x, self.support_y, self.query_x, self.query_y, self.support_m, self.query_m))



        return losses


    def predict_category(self, resuse=True):
        weights = self.weights
        support_x, support_y, query_x, query_y = self.support_x, self.support_y, self.query_x, self.query_y
        output_s = self.forward(support_x, weights, reuse=resuse)
        output_q = self.forward(query_x, weights, reuse=resuse)
        predict = category_choose(output_q, output_s, support_y)

        return predict
    def test_acc(self, inp, resuse=True):
        with tf.name_scope("test_acc"):
            weights = self.weights
            support_x, support_y, query_x, query_y = inp
            # output_s = self.forward(self.support_x, weights, reuse=resuse)
            # output_q = self.forward(self.query_x, weights, reuse=resuse)
            output_s = self.forward(support_x, weights, reuse=resuse)
            output_q = self.forward(query_x, weights, reuse=resuse)
            predict = category_choose(output_q, output_s,  support_y)
            accurcy = get_acc(predict, query_y)


        return accurcy


    def construct_model(self, input_tensor=None):
        if input_tensor is None:
            with tf.name_scope("input"):
                self.support_x = tf.placeholder(tf.float32, name='support_x')
                self.support_y = tf.placeholder(tf.float32, name='support_y')
                self.query_x = tf.placeholder(tf.float32, name='query_x')
                self.query_y = tf.placeholder(tf.float32, name='query_y')
                self.query_m = tf.placeholder(tf.float32, name='query_m')
                self.support_m = tf.placeholder(tf.float32, name='support_m')
        else:
            self.support_m = input_tensor['support_set'][2]
            self.support_x = input_tensor['support_set'][0]
            self.support_y = input_tensor['support_set'][1]
            self.query_m = input_tensor['query_set'][2]
            self.query_x = input_tensor['query_set'][0]
            self.query_y = input_tensor['query_set'][1]

        if 'weights' in dir(self):
            weights = self.weights
        else:
            self.weights = weights = self.construct_weights()
        accurcy = []
        losses = []
    def testop(self, batchinp):
        # batch_support_x, batch_support_y, batch_query_x, batch_query_y = batchinp
        batchinp = self.support_x, self.support_y, self.query_x, self.query_y
        # batch_support_x = tf.cast(tf.convert_to_tensor(batchinp[0]), dtype=tf.float32)
        # batch_support_y = tf.cast(tf.convert_to_tensor(batchinp[1]), dtype=tf.float32)
        # batch_query_x = tf.cast(tf.convert_to_tensor(batchinp[2]), dtype=tf.float32)
        # batch_query_y = tf.cast(tf.convert_to_tensor(batchinp[3]), dtype=tf.float32)
        # loss, acc = tf.map_fn(fn=self.get_loss, elems=(batch_support_x, batch_support_y, batch_query_x, batch_query_y), dtype=tf.float32, parallel_iterations=30)
        acc = tf.map_fn(fn=self.test_acc, elems=(batchinp), dtype=tf.float32, parallel_iterations=FLAGS.test_batch_size)

        # return tf.reduce_sum(acc)

        return acc



    def trainop(self):
        losses, acc = self.get_loss((self.support_x, self.support_y, self.query_x, self.query_y, self.support_m, self.query_m))

        if FLAGS.train:
            with tf.name_scope("loss"):
                # self.loss = loss = tf.reduce_sum(losses) / tf.to_float(FLAGS.query_num * FLAGS.model * FLAGS.way_num)
                self.loss = loss = tf.reduce_mean(losses)
            with tf.name_scope("compute_grad"):
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                # optimizer = tf.train.GradientDescentOptimizer(self.lr)
                # optimizer = tf.train.AdadeltaOptimizer(self.lr)


                self.gvs = gvs = optimizer.compute_gradients(loss)
                with tf.name_scope("clip_grad"):
                    # if FLAGS.data_source == 'PACS':
                    gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs]
                self.train = optimizer.apply_gradients(gvs)
            self.acc = acc
        return self.train, [self.loss, acc]








    def vectorlize(self, x):
        with tf.name_scope('vectorlize'):
            size = math.floor(FLAGS.image_size/(2**4))
            x = tf.reshape(tensor=x, shape=[-1, size*size*FLAGS.filter_num])
        return x

    def construct_conv(self):
        weights = {}
        dtype = tf.float32
        conv_initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=dtype, seed=0)
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
            hidden1 = conv_block(inp, weights['conv1'], weights['b1'], scope+'1', reuse)
            hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], scope+'2',reuse)
            hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], scope+'3',reuse)
            hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], scope+'4',reuse)
        return hidden4


    def construct_res(self):
        pass

    def forward_res(self, inp, weights, reuse=False):


        pass



