#-*-coding:utf-8-*-
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import tensorflow as tf
import numpy as np
import cv2
import os
import math
FLAGS = flags.FLAGS
np.random.seed(0)

def config(data_source):
    configs = {}
    if data_source == 'PACS':
        configs['PATH'] = FLAGS.data_PATH
        configs['split_txt_PATH'] = FLAGS.split_txt_PATH
        configs['model'] = ['art_painting', 'cartoon', 'photo', 'sketch']
        configs['split'] = ['train', 'test']
    elif data_source == 'mini-imagenet':
        configs['PATH'] = FLAGS.data_PATH
        configs['split_txt_PATH'] = FLAGS.split_txt_PATH
        configs['model'] = ['photo', 'sketch']
        configs['split'] = ['train', 'test', 'val']
    return configs.values()


def readtxt(filename, image_PATH):
    """
    :param filename: .txt
    :return: context of .txt
    """
    nameList = []
    txt = open(filename, 'r')
    for line in txt:
        item = {'path':"", 'label':0}
        # print(line)
        item['path'], item['label'] = line.split()
        item['path'] = os.path.join(image_PATH, item['path'])
        nameList.append(item)
    txt.close()
    return nameList
def readcsv(filename, image_PATH):
    nameList = []
    csv = open(filename, 'r')
    for line in csv:
        item = {'path': "", 'label': 0}
        # print(line)
        line = line.strip('\n')
        item['path'], item['label'] = line.split(',')
        item['path'] = os.path.join(image_PATH, item['path'])
        nameList.append(item)
    csv.close()
    return nameList[1:]

def get_split(train, test):
    """

    :param train: txt
    :param test: txt
    :return:data
    """
    data = {'train':{}, 'text':{}}
    if FLAGS.data_source == 'PACS':
        read_split = readtxt
    elif FLAGS.data_source == 'mini-imagenet':
        read_split = readcsv
    data['train'] = read_split(train)
    data['test'] = read_split(test)

    return data

def groupByLabel(data, class_num):
    """

    :param data: list producted by readtxt(PATH)
    :param class_num: num of classes
    :return: list for each classes
    """
    group = [[] for i in range(class_num)]
    for item in data:
        group[int(item['label'])-1].append(item['path'])
    return group

def split(model, split_PATH, train,image_PATH = "/data2/hsq/Project/PACS"):
    """
    :param PATH: root dir of split .txt file.
    :return: file name of data group by label each model e.g. {'art_painting':list[] .....}
    """
    if train:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    splits = os.listdir(split_PATH)
    trainsplits = [os.path.join(split_PATH, s) for s in splits if train_or_test in s]
    if FLAGS.data_source == 'PACS':
        dic_model_file = {m: s for m in model for s in trainsplits if m in s}
        data_model = {m: groupByLabel(readtxt(dic_model_file[m], image_PATH), 7) for m in model}
    # elif FLAGS.data_source == 'mini-imagenet':
    #     dic_model_file = {m: s for m in model for s in trainsplits if m in s}
    #     path_label = readcsv(dic_model_file[m], image_PATH)
    #     data_model = {m: groupByLabel(readcsv(dic_model_file[m], image_PATH), 7) for m in model}
    return data_model

def sample_support(support_num=5):
    pass

# def sample_task(data_model=split(), query_num_per_class_per_model=1, class_num=5, support_num_per_class_per_model=1):
def sample_task(query_num_per_class_per_model=1, class_num=5,
                    support_num_per_class_per_model=1, train=True):
    """
    :param data_model: list of module name e.g. ['art_painting', 'cartoon', 'photo', 'sketch']
    :param class_num: n-ways.
    :return: list of dict of path and label e.g [{'support_xy': [['/data2/hsq/Project/metric_PACS/pacs_filename/art_painting/horse/pic_072.jpg'], [1]],
                                                'q_xy': [['/data2/hsq/Project/metric_PACS/pacs_filename/art_painting/horse/pic_091.jpg'], [1]]]

    """
    if FLAGS.data_source == 'PACS':
        raw_path, split_txt, model, train_test = config('PACS')
        data_model = split(model, split_txt, train, raw_path)
        raw_class_num = 7
    elif FLAGS.data_source == 'mini-imagenet':
        raw_path, split_txt, model, train_test = config('mini-imagenet')
        data_model, raw_class_num = split_imagenet(model, split_txt, train, raw_path)

    classes = []
    task_data = []
    while True:
        n = np.random.randint(0, raw_class_num-1)
        if n not in classes:
            classes.append(n)
        if len(classes) == class_num: break
    s = {'data':[], 'label':[], 'modal':[]}
    q = {'data':[], 'label':[], 'modal':[]}
    for m_label, m in enumerate(model):
        for i, c in enumerate(classes):
            support_x = []
            support_y = []
            support_modal = []
            query_x = []
            query_y = []
            query_modal = []
            while True:
                idx = np.random.randint(len(data_model[m][c]))
                filename = data_model[m][c][idx]
                if filename not in support_x:
                    support_x.append(filename)
                    support_y.append(i)
                    support_modal.append(m_label)
                if(len(support_x) == support_num_per_class_per_model): break
            while True:
                idx = np.random.randint(len(data_model[m][c]))
                filename = data_model[m][c][idx]
                if (filename not in support_x) and (filename not in query_x):
                    query_x.append(filename)
                    query_y.append(i)
                    query_modal.append(m_label)
                if(len(query_x) == query_num_per_class_per_model): break

            s['data'].extend(support_x)
            s['label'].extend(support_y)
            s['modal'].extend(support_modal)
            q['data'].extend(query_x)
            q['label'].extend(query_y)
            q['modal'].extend(query_modal)
    return {'support': s, 'query': q}
def split_imagenet(model, split_txt, train, raw_path):
    if train:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    split_csv_path = [os.path.join(split_txt, t) for t in os.listdir(split_txt) if train_or_test in t][0]
    fliename_label = [readcsv(split_csv_path, os.path.join(raw_path, m)) for m in model]
    labels0 = set([fl['label'] for fl in fliename_label[1]])
    class_num = len(labels0)
    total_group = {m: [] for m in model}
    for k, m in enumerate(model):
        group = [[] for i in range(class_num)]
        for i, c in enumerate(labels0):
            group[i] = [fl['path'] for fl in fliename_label[k] if c == fl['label']]
        total_group[m] = group
    return total_group, class_num




def make_set_tensor(dict_set):
    """
    :param dic_set: data_dict e.g. support_set={'data': [...], 'label':[...], 'modal':[...].}
    :return: image tensors and label-one-hot tensors.
    """
    file_name_list = dict_set['data']
    label_list = dict_set['label']
    modal_list = dict_set['modal']
    m_modal = max(modal_list) + 1
    n_ways = max(label_list) + 1
    labels = np.eye(n_ways)[label_list]
    modals = np.eye(m_modal)[modal_list]
    img_batch = np.array([cv2.resize(cv2.imread(p),(FLAGS.image_size, FLAGS.image_size)) for p in file_name_list]).astype(np.float)/255.0
    index = np.arange(img_batch.shape[0])
    #shuffle the sample.
    np.random.shuffle(index)
    tmp_img = np.array([img_batch[i] for i in index])
    tmp_label = np.array([labels[i] for i in index])
    tmp_modal = np.array([modals[i] for i in index])
    return tmp_img, tmp_label.astype(np.float), tmp_modal.astype(np.float)





## Network helpers
##tf.nn.leaky_relu
def conv_block(inp, cweight, bweight, scope, reuse, activation=tf.nn.leaky_relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]
    with tf.name_scope("conv"):
        if FLAGS.max_pool:
            conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
        else:
            conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
        normed = normalize(conv_output, activation, scope, reuse)
        if FLAGS.max_pool:
            normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed
# tf.add
def normalize(inp, activation, scope, reuse):
    with tf.name_scope("normalize"):
        if FLAGS.norm == 'batch_norm':
            return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        elif FLAGS.norm == 'layer_norm':
            return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        elif FLAGS.norm == 'None':
            if activation is not None:
                return activation(inp)
            else:
                return inp


# loss function define
def distance(x, y, sw=None):
    """different distance function."""
    with tf.name_scope("compute_distance"):
        if sw == 'weight':
            width = FLAGS.way_num
            high = FLAGS.support_num * FLAGS.model * FLAGS.way_num
        elif FLAGS.prototype:
            if sw == 'class_prototype':
                width = FLAGS.way_num
            else:
                width = FLAGS.way_num * FLAGS.model
            high = FLAGS.query_num * FLAGS.model * FLAGS.way_num
        else:
            width = FLAGS.support_num * FLAGS.model * FLAGS.way_num
            high = FLAGS.query_num * FLAGS.model * FLAGS.way_num
        def inner_product(x, y):
            with tf.name_scope("inner_product"):
                res = tf.matmul(x, tf.transpose(y))
            return res
        def cosine(x, y):
            with tf.name_scope("cosine"):
                x_2 = tf.reshape(1/tf.sqrt(tf.reduce_sum(x * x, axis=1)), (-1, 1))
                y_2 = tf.reshape(1/tf.sqrt(tf.reduce_sum(y * y, axis=1)), (1, -1))
                x_fill_op = tf.ones((1, width), dtype=tf.float32)
                y_fill_op = tf.ones((high, 1), dtype=tf.float32)
                ip = tf.matmul(x, tf.transpose(y))
                res = tf.matmul(x_2, x_fill_op) * tf.matmul(y_fill_op, y_2) * ip
            return res
        if FLAGS.distance_style == 'euc_v1':
            with tf.name_scope('euc_v1'):
                x_2 = tf.reshape(tf.reduce_sum(x * x, axis=1), (-1, 1))
                y_2 = tf.reshape(tf.reduce_sum(y * y, axis=1), (1, -1))
                x_fill_op = tf.ones((1, width), dtype=tf.float32)
                y_fill_op = tf.ones((high, 1), dtype=tf.float32)
                xy = 2.0 * tf.matmul(x, tf.transpose(y))
                res = tf.matmul(x_2, x_fill_op) + tf.matmul(y_fill_op, y_2) - xy
                distance = tf.sqrt(tf.nn.relu(tf.matmul(x_2, x_fill_op) + tf.matmul(y_fill_op, y_2) - xy) + tf.ones_like(res)*0.000001)
        elif FLAGS.distance_style == 'cosine':
            distance = -cosine(x, y)
        elif FLAGS.distance_style == 'inner_product':
            distance = inner_product(x, y)
        # print("distance shape is:", distance.shape)
    return distance
def scd(support_x, query_x, s_label, q_label):
    with tf.name_scope("same_class_distance"):
        querys_to_supports_dist = distance(query_x, support_x)
        same_label_sifer = tf.matmul(q_label, tf.transpose(s_label))
        same_label_dist = tf.reduce_sum(querys_to_supports_dist * same_label_sifer, axis=1)
        mean_matrix = tf.reduce_sum(same_label_sifer, axis=1)
        res = same_label_dist/mean_matrix
    return res

def support_label_modal_proto(support_x, support_y, support_m):
    with tf.name_scope("support_label_modal_prototype"):
        sifter = tf.map_fn(lambda sm: tf.matmul(tf.reshape(sm, (-1, 1)), tf.ones((1, 5)))*support_y, elems=tf.transpose(support_m),
                           dtype=tf.float32, parallel_iterations=FLAGS.model)
        modal_proto_type = tf.map_fn(lambda x: tf.matmul(tf.diag(1/tf.reduce_sum(x, axis=0)), tf.matmul(tf.transpose(x), support_x)), elems=sifter,
                                     dtype=tf.float32, parallel_iterations=FLAGS.model)
        modal = tf.map_fn(lambda m: tf.matmul(tf.ones((5, 1)), tf.reshape(m, (1, -1))), elems=tf.diag(tf.ones((FLAGS.model))), dtype=tf.float32, parallel_iterations=FLAGS.model)
        label = tf.tile(tf.diag(tf.ones_like(support_y[0])), (FLAGS.model, 1))
    return tf.reshape(modal_proto_type, (FLAGS.way_num * FLAGS.model, -1)), \
           tf.reshape(modal, (FLAGS.way_num * FLAGS.model, -1)), \
           tf.reshape(label, (FLAGS.way_num * FLAGS.model, -1))
def get_prototype(support_x, s_label):
    with tf.name_scope("compute_prototype"):
        mean_matrix = tf.diag(1/tf.reduce_sum(s_label, axis=0))
        prototypes = tf.matmul(mean_matrix, tf.matmul(tf.transpose(s_label), support_x))
    return prototypes

def support_weight(support_x, s_label):
    with tf.name_scope("support_weight"):
        prototypes = get_prototype(support_x, s_label)
        sample_to_proto_dist = tf.reshape(tf.exp(tf.reduce_sum(-distance(support_x, prototypes, sw=True) * (2*s_label-tf.ones_like(s_label)), axis=1)), (-1, 1))
        class_sum = tf.reshape(tf.transpose(tf.matmul(tf.transpose(s_label), sample_to_proto_dist)), (-1, 1))
        sum_up = tf.matmul(s_label, class_sum)
        weights = sample_to_proto_dist / sum_up
    return weights * support_x


def category_choose(output_q, output_s, label_s):
    with tf.name_scope('category_choose'):
        with tf.name_scope('category'):
            if FLAGS.prototype:
                prototypes = get_prototype(output_s, label_s)
                label_s = tf.diag(tf.ones(FLAGS.way_num))
                softmax = get_dist_category(output_q, prototypes, label_s)
            else:
                softmax = get_dist_category(output_q, output_s, label_s)
    return softmax

def loss_eps_p(support_x, query_x, s_modal, q_modal, s_label, q_label, margin):
    with tf.name_scope("loss_eps_prototype"):
        modal_proto_type, p_modal, p_label = support_label_modal_proto(support_x, s_label, s_modal)
        querys_to_proto_dist = distance(query_x, modal_proto_type)
        def sifter(matrix_query, matrix_support):
            return tf.matmul(matrix_query, tf.transpose(matrix_support))
        # choose the same modal and same label.
        modal_sifter = sifter(q_modal, p_modal)
        label_sifter = sifter(q_label, p_label)
        same_label_modal = modal_sifter * label_sifter
        # apply on the distance matrix.
        same_label_diff_model_sifter = label_sifter - same_label_modal
        if not FLAGS.eps_usehard:
            SLDMS_mean = tf.reduce_sum(same_label_diff_model_sifter, axis=1)
            # print(tf.reduce_sum(same_label_diff_model_sifter * querys_to_supports_dist, axis=1).eval())
            same_label_diff_model_dist = tf.reduce_sum(
                tf.multiply(same_label_diff_model_sifter, querys_to_proto_dist), axis=1) / SLDMS_mean
            same_modal_diff_label_dist = (modal_sifter - same_label_modal) * querys_to_proto_dist
            mean_matrix = tf.matmul((modal_sifter - same_label_modal), p_label)
            mean_matrix = (mean_matrix + 0.00000000001 * tf.ones_like(mean_matrix))
            group_by_label = tf.matmul(same_modal_diff_label_dist, p_label) / mean_matrix
        else:
            same_label_diff_model_dist = tf.reduce_max(same_label_diff_model_sifter * querys_to_proto_dist, axis=1)
            dist_smdl = (modal_sifter - same_label_modal) * querys_to_proto_dist
            min_idx = tf.cast(
                tf.reduce_sum(tf.matmul(tf.transpose(p_label), tf.transpose(modal_sifter - same_label_modal)), axis=1)[
                    0] / tf.reduce_sum(modal_sifter - same_label_modal, axis=1)[0], tf.int32)
            group_by_label = tf.map_fn(fn=lambda x: tf.sort(x * tf.transpose(p_label)), elems=dist_smdl,
                                       dtype=tf.float32,
                                       parallel_iterations=FLAGS.model * FLAGS.way_num * FLAGS.query_num)[:, :,
                             -min_idx]
        margin_exp_GBL = tf.reduce_sum(tf.exp(-group_by_label + tf.ones_like(group_by_label) * margin), axis=1)
        margin_exp_GBL = margin_exp_GBL - tf.exp(margin * tf.ones_like(margin_exp_GBL))
        exp_SLDM = tf.exp(-same_label_diff_model_dist)
        loss_eps = -tf.log(exp_SLDM / (margin_exp_GBL + exp_SLDM))
    return loss_eps


def loss_eps(support_x, query_x, s_modal, q_modal, s_label, q_label, margin):
    with tf.name_scope("loss_eps"):
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
        if not FLAGS.eps_usehard:
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


def intra_dist(dist, weights, query_y, support_y):
    """1(query) v. n(support) compare."""
    with tf.name_scope("intra_dist"):
        with tf.name_scope("same_matrix"):
            qy = tf.cast(query_y, dtype=tf.bool)
            sy = tf.cast(support_y, dtype=tf.bool)
            same_matrix = tf.cast(tf.logical_and(qy, sy), dtype=tf.float32)
            sm = tf.matmul(weights, same_matrix)
        res = tf.reduce_sum(dist*sm)
    return res
def get_differ_matrix(qy, sy):
    return tf.cond(pred=tf.reduce_all(tf.equal(qy,sy)), true_fn=lambda:tf.zeros_like(qy), false_fn=lambda:sy)

def inter_dist(dist, weights, query_y, support_y, t):
    """1(query) v. n(support) compare."""
    with tf.name_scope("inter_dist"):
        with tf.name_scope("differ_matrix"):
            differ_matrix = tf.map_fn(fn=lambda sy:get_differ_matrix(query_y, sy), elems=support_y, parallel_iterations=FLAGS.support_num*FLAGS.way_num*FLAGS.model)
            dm = tf.matmul(weights, differ_matrix)
        res = dist*dm
        T = (-t)*dm
    return res+T


def get_acc(pred, actual):
    with tf.name_scope("compute_accu"):
        p = tf.cast(tf.one_hot(tf.arg_max(pred, 1), FLAGS.way_num), dtype=tf.bool)
        a = tf.cast(actual, dtype=tf.bool)
        acc = tf.reduce_sum(tf.cast(tf.logical_and(p, a),
                                 dtype=tf.float32))/\
           tf.cast(FLAGS.query_num*FLAGS.model*FLAGS.way_num, dtype=tf.float32)
    return acc


def get_dist_category(x, y, y_onehot_label):
    with tf.name_scope("compute_distance_onehot"):
        # if FLAGS.distance_style == 'euc':
        #     dist = tf.map_fn(fn=lambda x_: distance(x_, y), elems=x, dtype=tf.float32, parallel_iterations=FLAGS.model*FLAGS.way_num*FLAGS.query_num)
        # elif FLAGS.distance_style == 'cosine':
        dist = distance(x, y, 'class_prototype')
        res = tf.exp(-tf.matmul(dist, y_onehot_label))
        sum = tf.diag(1/tf.reduce_sum(res, axis=1))
        softmax = tf.matmul(sum, res)
    return softmax

def category_sifter(label):
    with tf.name_scope("category_sifter"):
        sifter = tf.transpose(label)
    return sifter
def sift_set(sx, sy):
    with tf.name_scope("mean_sifter_x"):
        sifter_y = category_sifter(sy)
        num_diag = tf.matrix_diag(1/tf.reduce_sum(sifter_y, axis=1))
        sifter_x = tf.matmul(sifter_y, sx)
        mean_x = tf.matmul(num_diag, sifter_x)
    return mean_x
def get_weights_diag_matrix(sx, sy):
    """get the weights for each sample, based on distance."""
    with tf.name_scope("get_sample_weights_matrix"):
        meanx = tf.matmul(sy, sift_set(sx, sy))
        dist = tf.reshape(tf.exp(-tf.reduce_sum(tf.square(sx - meanx), axis=1)), (1, -1))
        sums = 1/tf.matmul(dist, sy)
        distance_matrix = tf.matrix_diag(tf.matmul(sums, tf.transpose(sy)))
        weights = tf.matrix_diag(tf.matmul(dist, distance_matrix))
    return weights
def vectorlize(x):
    with tf.name_scope('vectorlize'):
        size = math.floor(FLAGS.image_size/(2**4))
        x = tf.reshape(tensor=x, shape=[-1, size*size*FLAGS.filter_num])
    return x


## Loss functions
def mse(pred, label):
    return tf.reduce_mean(tf.square(pred-label), axis=1)
def log_liklyhood(pred, label):
    with tf.name_scope('log_category_loss'):
        # res = tf.reduce_sum(-tf.log(tf.reduce_sum(pred * label, axis=1)))
        res = -tf.log(tf.reduce_sum(pred * label, axis=1))

    return res


def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
def category_loss(predict, query_y):
    with tf.name_scope('mse_category_loss'):
        loss = mse(predict, query_y)
    return loss


def inter_var(support_x, s_label):
    with tf.name_scope("intra_var"):
        prototypes = get_prototype(support_x, s_label)
        center_point = tf.reduce_mean(prototypes, axis=0)
        vars = tf.reduce_mean(tf.sqrt(tf.square(center_point-prototypes)))
    return vars
def intra_var(support_x, s_label):
    with tf.name_scope("intra_var"):
        prototypes = get_prototype(support_x, s_label)
        vars = tf.reduce_mean(tf.reduce_sum(distance(support_x, prototypes, sw=True) * s_label, axis=1))
    return vars

def compute_loss(qxy, support_x, support_y, t=1.0):
    """comput distance based loss.
       qxy is a tuple:(query_x, query_y).
       q(1) v. s(n).
    """
    with tf.name_scope("distance_loss"):
        query_x, query_y = qxy
        weights = get_weights_diag_matrix(support_x, support_y)
        dist = distance(query_x, support_x)
        intrad = intra_dist(dist, weights, query_y, support_y)
        interd = inter_dist(dist, weights, query_y, support_y, t)
        log_likely_hood = -tf.log(tf.exp(-intrad)/(tf.exp(-intrad) + tf.reduce_sum(tf.exp(-interd))))
    return log_likely_hood
    # return interd
