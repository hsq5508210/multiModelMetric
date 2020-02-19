#-*-coding:utf-8-*-
import tensorflow as tf
from utils import compute_loss
from tensorflow.python.platform import flags
import tensorflow as tf
import numpy as np
from time import time
np.random.seed(0)
query_set = np.random.normal(size=(10, 3))
s_modal = tf.convert_to_tensor([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]], dtype=tf.float32)
q_modal = tf.convert_to_tensor([[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0],[0,1]], dtype=tf.float32)
s_label = tf.convert_to_tensor([[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1]], dtype=tf.float32)
q_label = tf.convert_to_tensor([[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1]], dtype=tf.float32)
q = np.random.normal(size=(1, 3))
q = tf.convert_to_tensor(q, dtype=tf.float32)
support_set = np.random.normal(size=(20, 3))
query_num = 10
support__num = 20
vector_dim = 3
q_test = tf.convert_to_tensor(np.random.normal(size=(query_num, vector_dim)), dtype=tf.float32)
s_test = tf.convert_to_tensor(np.random.normal(size=(support__num, vector_dim)), dtype=tf.float32)
distance_style = 'euc_v1'
usehard = False

def get_dist_category(x, y, y_onehot_label):
    with tf.name_scope("compute_distance_onehot"):
        # if FLAGS.distance_style == 'euc':
        #     dist = tf.map_fn(fn=lambda x_: distance(x_, y), elems=x, dtype=tf.float32, parallel_iterations=FLAGS.model*FLAGS.way_num*FLAGS.query_num)
        # elif FLAGS.distance_style == 'cosine':
        dist = distance(x, y)
        res = tf.exp(-tf.matmul(dist, y_onehot_label))
        sum = tf.diag(1/tf.reduce_sum(res, axis=1))
        softmax = tf.matmul(sum, res)
    return softmax
def distance(x, y, style=distance_style):
    if style == 'euc':
        return tf.sqrt(tf.reduce_sum(tf.square(x-y), axis=1))
    if style == 'euc_v1':
        return distance_v1(x, y)
    elif style == 'cosine':
        return -cosine(x, y)

def distance_v1(x, y):
    with tf.name_scope('euc_v1'):
        x_2 = tf.reshape(tf.reduce_sum(x * x, axis=1), (-1, 1))
        y_2 = tf.reshape(tf.reduce_sum(y * y, axis=1), (1, -1))
        width = y.shape[0]
        high = x.shape[0]
        x_fill_op = tf.ones((1, width), dtype=tf.float32)
        y_fill_op = tf.ones((high, 1), dtype=tf.float32)
        xy = 2.0 * tf.matmul(x, tf.transpose(y))
        res = tf.sqrt(tf.matmul(x_2, x_fill_op) + tf.matmul(y_fill_op, y_2) - xy)
    return res

def compare_dist(support_x, query_x):
    s1 = time()
    dist_map = tf.map_fn(fn=lambda q: distance(q, support_x, 'euc'), elems=query_x, dtype=tf.float32, parallel_iterations=query_num)
    s2 = e1 = time()
    dist_v1 = distance(query_x, support_x, 'euc_v1')
    s3 = e2 = time()
    dist_cosine = distance(query_x, support_x, 'cosine')
    e3 = time()
    print("dist map shape is:", dist_map.shape, "spent ", str(e1-s1), 's')
    print("dist v1 shape is:", dist_v1.shape, "spent ", str(e2-s2), 's')
    print("dist cosine shape is:", dist_v1.shape, "spent ", str(e3 - s3), 's')

def inner_product(x, y):
    return tf.matmul(x, tf.transpose(y))
def cosine(x, y):
    l2_x = tf.diag(tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)))
    l2_y = tf.diag(tf.sqrt(tf.reduce_sum(tf.square(y), axis=1)))
    ip = inner_product(x, y)
    return tf.matmul(tf.matmul(l2_x, ip), l2_y)

def get_acc(pred, actual):
    with tf.name_scope("compute_accu"):
        p = tf.cast(tf.one_hot(tf.arg_max(pred, 1), 5), dtype=tf.bool)
        print(p.eval())
        a = tf.cast(actual, dtype=tf.bool)
        print(a.eval())
        acc = tf.reduce_sum(tf.cast(tf.logical_and(p, a),
                                 dtype=tf.float32))/\
           tf.cast(10, dtype=tf.float32)
    return acc

query_x = tf.cast(tf.convert_to_tensor(query_set), dtype=tf.float32)
q_sample = tf.cast(tf.convert_to_tensor(q), dtype=tf.float32)
support_x = tf.cast(tf.convert_to_tensor(support_set), dtype=tf.float32)

def vectorlize(x):
    with tf.name_scope('vectorlize'):
        x = tf.reshape(tensor=x, shape=[-1, 3])
    return x
def mse(pred, label):
    return tf.reduce_mean(tf.square(pred-label), axis=1)
def category_loss(predict, query_y):
    with tf.name_scope('category_loss'):
        loss = mse(predict, query_y)
    return loss
def loss_eps(support_x, query_x, s_modal, q_modal, s_label, q_label, margin):
    with tf.name_scope("loss_eps"):
        support_x = vectorlize(support_x)
        query_x = vectorlize(query_x)
        # if FLAGS.distance_style == 'euc':
        #     querys_to_supports_dist = tf.map_fn(fn=lambda q: distance(q, support_x), elems=query_x, dtype=tf.float32)
        # elif FLAGS.distance_style == 'cosine':
        querys_to_supports_dist = distance(query_x, support_x)
        def sifter(matrix_query, matrix_support):
            return tf.matmul(matrix_query, tf.transpose(matrix_support))
        # choose the same modal and same label.
        modal_sifter = sifter(q_modal, s_modal)
        label_sifter = sifter(q_label, s_label)
        same_label_modal = modal_sifter * label_sifter
        # apply on the distance matrix.
        same_label_diff_model_sifter = label_sifter - same_label_modal
        if not usehard:
            SLDMS_mean = tf.reduce_sum(same_label_diff_model_sifter, axis=1)
            # print(tf.reduce_sum(same_label_diff_model_sifter * querys_to_supports_dist, axis=1).eval())
            same_label_diff_model_dist = tf.reduce_sum(tf.multiply(same_label_diff_model_sifter, querys_to_supports_dist), axis=1) / SLDMS_mean
            same_modal_diff_label_dist = (modal_sifter-same_label_modal) * querys_to_supports_dist
            mean_matrix = tf.matmul((modal_sifter-same_label_modal), s_label)
            mean_matrix = (mean_matrix + 0.00000000001*tf.ones_like(mean_matrix))
            group_by_label = tf.matmul(same_modal_diff_label_dist, s_label) / mean_matrix
        else:
            same_label_diff_model_dist = tf.reduce_max(same_label_diff_model_sifter * querys_to_supports_dist, axis=1)
            dist_smdl = (modal_sifter-same_label_modal) * querys_to_supports_dist
            min_idx = tf.cast(tf.reduce_sum(tf.matmul(tf.transpose(s_label), tf.transpose(modal_sifter-same_label_modal)), axis=1)[0]/ tf.reduce_sum(modal_sifter-same_label_modal, axis=1)[0], tf.int32)
            group_by_label = tf.map_fn(fn=lambda x: tf.sort(x * tf.transpose(s_label)), elems=dist_smdl, dtype=tf.float32, parallel_iterations=FLAGS.model * FLAGS.way_num * FLAGS.query_num)[:, :, -min_idx]
        margin_exp_GBL = tf.reduce_sum(tf.exp(-group_by_label + tf.ones_like(group_by_label) * margin), axis=1)
        margin_exp_GBL = margin_exp_GBL - tf.exp(margin * tf.ones_like(margin_exp_GBL))
        exp_SLDM = tf.exp(-same_label_diff_model_dist)
        loss_eps = -tf.log(exp_SLDM / (margin_exp_GBL + exp_SLDM))
    return loss_eps


def scd(support_x, query_x, s_label, q_label):
    with tf.name_scope("same_class_distance"):
        querys_to_supports_dist = distance(query_x, support_x)
        same_label_sifer = tf.matmul(q_label, tf.transpose(s_label))
        same_label_dist = tf.reduce_sum(querys_to_supports_dist * same_label_sifer, axis=1)
        mean_matrix = tf.reduce_sum(same_label_sifer, axis=1)
        res = same_label_dist/mean_matrix
    return res
def get_prototype(support_x, s_label):
    with tf.name_scope("get_prototype"):
        mean_matrix = tf.diag(1/tf.reduce_sum(s_label, axis=0))
        prototypes = tf.matmul(mean_matrix, tf.matmul(tf.transpose(s_label), support_x))
    return prototypes
def support_weight(support_x, s_label):
    with tf.name_scope("support_weight"):
        prototypes = get_prototype(support_x, s_label)
        sample_to_proto_dist = tf.reshape(tf.exp(tf.reduce_sum(-distance_v1(support_x, prototypes) * (2*s_label-tf.ones_like(s_label)), axis=1)), (-1, 1))
        class_sum = tf.reshape(tf.transpose(tf.matmul(tf.transpose(s_label), sample_to_proto_dist)), (-1, 1))
        sum_up = tf.matmul(s_label, class_sum)
        weights = sample_to_proto_dist / sum_up
    return weights * support_x




with tf.Session() as sess:
    w = support_weight(s_test, s_label)
    print(w.eval())
    # cos = sess.run(cosine(query_x, support_x))
    # print(loss1-loss2)

