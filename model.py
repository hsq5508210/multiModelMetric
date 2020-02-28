#-*-coding:utf-8-*-

import tensorflow as tf
import math
from tensorflow.python.platform import flags
from utils import conv_block, get_acc, loss_eps, category_choose, category_loss, vectorlize, log_liklyhood, scd, support_weight, intra_var, inter_var, loss_eps_p, proto_various_loss
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
        if FLAGS.backbone == 'Conv64F':
            self.construct_weights = self.construct_conv
            self.forward = self.forward_conv
        else:
            self.construct_weights = self.construct_res
            self.forward = self.forward_res
        if FLAGS.loss_function == 'mse':
            self.loss_function = category_loss
        elif FLAGS.loss_function == 'log':
            self.loss_function = log_liklyhood
        elif FLAGS.loss_function == 'sce':
            self.loss_function = tf.losses.softmax_cross_entropy
        if FLAGS.prototype:
            self.eps = loss_eps_p
        else:
            self.eps = loss_eps


    def decay(self, i):
        iter = FLAGS.decay_iteration
        decay_rate = FLAGS.decay_rate
        self.lr = self.lr * (decay_rate ** float(i//iter))
        # self.lr = tf.train.exponential_decay(self.lr, global_step=global_step, decay_steps=decay_steps,
        #                                       decay_rate=decay_rate)

    def get_loss(self, inp, resuse=True, dist_weight=0.2):
        with tf.name_scope("compute_loss"):
            weights = self.weights
            support_x, support_y, query_x, query_y, support_m, query_m = inp
            if FLAGS.adapt_prototypes:
                fast_weights = self.adapt_prototypes((support_x, support_y, support_m))
                output_s = vectorlize(self.forward(support_x, fast_weights, reuse=resuse))
                output_q = vectorlize(self.forward(query_x, fast_weights, reuse=resuse))
            else:
                output_s = vectorlize(self.forward(support_x, weights, reuse=resuse))
                output_q = vectorlize(self.forward(query_x, weights, reuse=resuse))
            if FLAGS.support_weight:
                output_s = output_s + support_weight(output_s, support_y)

            self.predict = predict = category_choose(output_q, output_s,  support_y)
            accurcy = get_acc(predict, query_y)

            # task_losses = self.loss_function(predict, query_y)
            task_losses = self.loss_function(query_y, predict)

            if FLAGS.intra_var:
                task_losses = task_losses + intra_var(output_s, support_y) * 0.1 * tf.ones(shape=(FLAGS.model * FLAGS.way_num * FLAGS.query_num))
                if FLAGS.inter_var:
                    task_losses = task_losses + (intra_var(output_s, support_y)/inter_var(output_s, support_y)) * 0.1 * tf.ones(shape=(FLAGS.model * FLAGS.way_num * FLAGS.query_num))
            if FLAGS.eps_loss and FLAGS.category_loss:
                self.losses_eps = losses_eps = self.eps(output_s, output_q, support_m, query_m, support_y,
                                      query_y, FLAGS.margin)
                losses = (1 - self.w) * task_losses + self.w * losses_eps
                if not FLAGS.same_class_dist:
                    return losses, accurcy
                else:
                    same_class_dist = scd(output_s, output_q, support_y, query_y)
                    # losses = (1 - self.w) * task_losses + (self.w - dist_weight) * losses_eps + dist_weight * same_class_dist
                    losses = (1 - self.w) * task_losses + (self.w) * losses_eps + dist_weight * same_class_dist
                    return losses, accurcy
            elif FLAGS.category_loss:
                return task_losses, accurcy
            elif FLAGS.eps_loss:
                self.losses_eps = losses_eps = loss_eps(output_s, output_q, support_m, query_m, support_y,
                                                        query_y, FLAGS.margin)
                return losses_eps, accurcy

    def adapt_prototypes(self, inp):
        support_x, support_y, support_m = inp
        weights = self.weights
        output_s = vectorlize(self.forward(support_x, weights, reuse=True))
        for i in range(FLAGS.finetune_times):
            proto_var_loss = proto_various_loss(output_s, support_y, support_m)
            grads = tf.gradients(proto_var_loss, list(weights.values()))
            grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(weights.keys(), grads))
            fast_weights = dict(
                zip(weights.keys(), [weights[key] - FLAGS.finetune_lr * gradients[key] for key in weights.keys()]))
            output_s = vectorlize(self.forward(support_x, fast_weights, reuse=True))
        return fast_weights

    def debuf_nan(self, resuse=True):
        # dist = distance(output_q, output_s)
        # res = tf.exp(-tf.matmul(dist, self.support_y))
        # sum = tf.diag(1/tf.reduce_sum(res, axis=1))
        # softmax = tf.matmul(sum, res)

        # predict = category_choose(output_q, output_s, self.support_y)
        # task_losses = self.loss_function(predict, self.query_y)
        # losses_eps = loss_eps(output_s, output_q, self.support_m, self.query_m, self.support_y,
        #                       self.query_y, FLAGS.margin)

        # losses, acc = self.get_loss((self.support_x, self.support_y, self.query_x, self.query_y, self.support_m, self.query_m))
        weights = self.weights
        support_x, support_y, query_x, query_y, support_m, query_m = self.support_x, self.support_y, self.query_x, self.query_y, self.support_m, self.query_m
        output_s = vectorlize(self.forward(support_x, weights, reuse=resuse))
        output_q = vectorlize(self.forward(query_x, weights, reuse=resuse))
        losses_eps = self.eps(output_s, output_q, support_m, query_m, support_y,
                              query_y, FLAGS.margin)
        self.predict = predict = category_choose(output_q, output_s, support_y)
        losses = self.loss_function(predict, query_y)

        return [losses, losses_eps]


    def predict_category(self, resuse=True):
        weights = self.weights
        support_x, support_y, query_x, query_y = self.support_x, self.support_y, self.query_x, self.query_y
        output_s = self.forward(support_x, weights, reuse=resuse)
        output_q = self.forward(query_x, weights, reuse=resuse)
        predict = category_choose(output_q, output_s, support_y)

        return predict
    def test_acc(self, inp, resuse=True):
        with tf.name_scope("test_acc"):

            support_x, support_y, query_x, query_y, support_m = inp
            if FLAGS.adapt_prototypes:
                weights = self.adapt_prototypes((support_x, support_y, support_m))
            else:
                weights = self.weights
            output_s = vectorlize(self.forward(support_x, weights, reuse=resuse))
            output_q = vectorlize(self.forward(query_x, weights, reuse=resuse))
            if FLAGS.support_weight:
                output_s = output_s + support_weight(output_s, support_y)
            predict = category_choose(output_q, output_s,  support_y)
            accurcy = get_acc(predict, query_y)
        return accurcy
    def test_loss(self, inp, resuse=True):
        with tf.name_scope("test_loss"):
            support_x, support_y, query_x, query_y, support_m = inp
            if FLAGS.adapt_prototypes:
                weights = self.adapt_prototypes((support_x, support_y, support_m))
            else:
                weights = self.weights
            output_s = vectorlize(self.forward(support_x, weights, reuse=resuse))
            output_q = vectorlize(self.forward(query_x, weights, reuse=resuse))
            if FLAGS.support_weight:
                output_s = output_s + support_weight(output_s, support_y)
            predict = category_choose(output_q, output_s,  support_y)
            loss = tf.reduce_mean(self.loss_function(query_y, predict))

        return loss
    def test_al(self, resuse=True):
        support_x, support_y, query_x, query_y, support_m = self.support_x, self.support_y, self.query_x, self.query_y, self.support_m

        if FLAGS.adapt_prototypes:
            weights = self.adapt_prototypes((support_x, support_y, support_m))
        else:
            weights = self.weights
        with tf.name_scope("test_acc_loss"):
            output_s = vectorlize(self.forward(support_x, weights, reuse=resuse))
            output_q = vectorlize(self.forward(query_x, weights, reuse=resuse))
            if FLAGS.support_weight:
                output_s = output_s + support_weight(output_s, support_y)
            predict = category_choose(output_q, output_s, support_y)
            loss = tf.reduce_mean(self.loss_function(query_y, predict))
            accurcy = get_acc(predict, query_y)
            al = [accurcy, loss]
        del weights
        return al



    def testop(self, ):

        acc, loss = self.test_al()
        return [acc, loss]

    def construct_model(self, input_tensor=None):
        # if FLAGS.load_ckpt:
        #     self.support_x = graph.get_tensor_by_name('support_x')
        #     self.support_y = graph.get_tensor_by_name('support_y')
        #     self.query_x = graph.get_tensor_by_name('query_x')
        #     self.query_y = graph.get_tensor_by_name('query_y')
        #     self.query_m = graph.get_tensor_by_name('query_m')
        #     self.support_m = graph.get_tensor_by_name('support_m')

        if input_tensor is None:
            with tf.name_scope("input"):
                self.support_x = tf.placeholder(tf.float32, name='support_x')
                self.support_y = tf.placeholder(tf.float32, name='support_y')
                self.query_x = tf.placeholder(tf.float32, name='query_x')
                self.query_y = tf.placeholder(tf.float32, name='query_y')
                self.query_m = tf.placeholder(tf.float32, name='query_m')
                self.support_m = tf.placeholder(tf.float32, name='support_m')
        else:
            with tf.name_scope("input"):
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




    def trainop(self):

        self.losses, self.acc = self.get_loss((self.support_x, self.support_y, self.query_x, self.query_y, self.support_m, self.query_m))
        # l = tf.is_nan(tf.reduce_sum(losses)).eval()
        losses = self.losses
        with tf.name_scope("loss"):
            self.loss = loss = tf.reduce_mean(losses)
        if FLAGS.train:
            with tf.name_scope("compute_grad"):
                if FLAGS.optimizer == 'adam':
                    # optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)
                elif FLAGS.optimizer == 'sgd':
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                self.gvs = gvs = optimizer.compute_gradients(loss)
                with tf.name_scope("clip_grad"):
                    # if FLAGS.data_source == 'PACS':
                    gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]
                self.train = optimizer.apply_gradients(gvs)
        return self.train, [self.acc, loss]


        #     self.acc = acc
        # return self.train, [self.loss, acc]








    def vectorlize(self, x):
        with tf.name_scope('vectorlize'):
            size = math.floor(FLAGS.image_size/(2**4))
            x = tf.reshape(tensor=x, shape=[-1, size*size*FLAGS.filter_num])
        return x

    def construct_conv(self, ):

        weights = {}
        convnames = ['conv1', 'conv2', 'conv3', 'conv4']
        biasnames = ['weights/b1', 'weights/b2', 'weights/b3', 'weights/b4']
        with tf.name_scope('weights'):

            # if FLAGS.load_ckpt:
            #     conv_nodes = {node.name:node for node in graph.as_graph_def().node if node.name in convnames}
            #     # print(conv_nodes)
            #     weights['conv1'] = graph.get_tensor_by_name('conv1:0')
            #     print(weights['conv1'].name)
            #     weights['b1'] = graph.get_tensor_by_name('weights/b1:0')
            #     weights['conv2'] = graph.get_tensor_by_name('conv2:0')
            #     weights['b2'] = graph.get_tensor_by_name('weights/b2:0')
            #     weights['conv3'] = graph.get_tensor_by_name('conv3:0')
            #     weights['b3'] = graph.get_tensor_by_name('weights/b3:0')
            #     weights['conv4'] = graph.get_tensor_by_name('conv4:0')
            #     weights['b4'] = graph.get_tensor_by_name('weights/b4:0')

                # weights['conv1'] = conv_nodes['conv1']
                # weights['b1'] = graph.get_tensor_by_name('weights/b1:0')
                # weights['conv2'] = conv_nodes['conv2']
                # weights['b2'] = graph.get_tensor_by_name('weights/b2:0')
                # weights['conv3'] = conv_nodes['conv3']
                # weights['b3'] = graph.get_tensor_by_name('weights/b3:0')
                # weights['conv4'] = conv_nodes['conv4']
                # weights['b4'] = graph.get_tensor_by_name('weights/b4:0')
            # else:
            dtype = tf.float32
            if FLAGS.init_style == 'xavier':
                conv_initializer = tf.contrib.layers.xavier_initializer(uniform=False, dtype=dtype, seed=0)
            elif FLAGS.init_style =='normal':
                conv_initializer = tf.random_normal_initializer(seed=0, mean=0.0, stddev=0.02)
            k = 3
            weights['conv1'] = tf.get_variable('conv1', [k, k, self.channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b1')
            weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b2')
            weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b3')
            weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
            weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]), name='b4')
        return weights


    def forward_conv(self, inp, weights, reuse=False):
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



