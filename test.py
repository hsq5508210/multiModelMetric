#-*-coding:utf-8-*-
import tensorflow as tf
from utils import compute_loss
from tensorflow.python.platform import flags
qy = [[1,0,0], [0,0,1]]
sy = [[0,0,1],[1,0,0],[0,1,0],[1,0,0], [0,1,0]]
qx = [[1.0, 2.2, 2.0, 1.0], [0.0, 0.2, 2.5, 1.0]]
sx = [[2.0, 2.0, 2.0,3.0], [4.0,1.0, 1.0, 2.0], [1.0,2.0, 0.0, 1.0], [5.0,0.0, 3.0, 1.0], [1.0,2.0, 0.5, 1.0]]
category = [i for i in range(3)]
# category = tf.cast(tf.convert_to_tensor(category), dtype=tf.int8)
qy = tf.cast(tf.convert_to_tensor(qy), dtype=tf.float32)
sy = tf.cast(tf.convert_to_tensor(sy), dtype=tf.float32)
qx = tf.cast(tf.convert_to_tensor(qx), dtype=tf.float32)
sx = tf.cast(tf.convert_to_tensor(sx), dtype=tf.float32)
flags.DEFINE_string("distance_style", default="euc", help="how to compute the distance.")
flags.DEFINE_integer("way_num", default=3, help="the number of classify ways.")

FLAGS = flags.FLAGS
def distance(x, y):
    """different distance function."""
    with tf.name_scope("compute_distance"):
        distance = tf.sqrt(tf.reduce_sum(tf.square(x-y), axis=1))
        d = tf.map_fn(fn=lambda s:tf.fill(value=s, dims=(3, )), elems=distance)
    return d
def aeqb(a, b):
    eqint = tf.reduce_sum(tf.cast(tf.equal(qy, sy), dtype=tf.float32))

    return  tf.equal(eqint, tf.reduce_sum(tf.ones_like(a)))

def get_differ_matrix(qy, sy):
    return tf.cond(pred=tf.reduce_all(tf.equal(qy,sy)), true_fn=lambda:tf.zeros_like(qy), false_fn=lambda:sy)
def inter_dist(dist, query_y, support_y):
    """1(query) v. n(support) compare."""
    with tf.name_scope("inter_dist"):
        with tf.name_scope("differ_matrix"):
            differ_matrix = get_differ_matrix(query_y, support_y)
        res = tf.reduce_sum(dist*differ_matrix)
    return res
def category_sifter(label):
    with tf.name_scope("category_sifter"):
        sifter = tf.transpose(label)
    return sifter
def sift_set(sx, sy):
    with tf.name_scope("mean_sifter_x"):
        sifter_y = category_sifter(sy)
        num_diag = tf.matrix_diag(1/tf.reduce_sum(sifter_y, axis=1))
        sifter_x = tf.matmul(sifter_y,sx)
        mean_x = tf.matmul(num_diag, sifter_x)
    return mean_x
def get_weights_diag_matrix(sx, sy):
    with tf.name_scope("get_sample_weights_matrix"):

        meanx = tf.matmul(sy, sift_set(sx, sy))
        dist = tf.reshape(tf.exp(-tf.reduce_sum(tf.square(sx - meanx), axis=1)), (1,-1))
        sum = 1/tf.matmul(dist, sy)
        distance_matrix = tf.matrix_diag(tf.matmul(sum, tf.transpose(sy)))

    return tf.matrix_diag(tf.matmul(dist, distance_matrix))

# dist = distance(qx, sx)

# with tf.Session() as sess:
#     print(tf.transpose(sy).eval())
#     print(sy.eval())
#
#     # print(category)
#     # category = tf.one_hot(category, depth=3, axis=0)
#     # print(category.eval())
#     print(sx.eval())
#     # print(tf.matrix_diag(1/tf.reduce_sum(sy, axis=0)).eval())
#     # print(sift_set(sx, sy).eval())
#     # print(tf.matmul(getweights(sx, sy),sx).eval())
#     inp = (qx, qy)
#     l = tf.convert_to_tensor([0,1], dtype=tf.int8)
#     # qx = tf.reshape(qx, (2,1,-1))
#     # qy = tf.reshape(qy, (2, 1, -1))
#     print(qx.shape, qy.shape)
#
#     loss = tf.map_fn(fn=lambda qxy: compute_loss((qxy[0], qxy[1]), sx, sy, 1), elems=(qx, qy), dtype=tf.float32, parallel_iterations=2)
#     print(loss.eval())
import os
def config(data_source):
    configs = {}
    if data_source == 'PACS':
        configs['PATH'] = FLAGS.data_PATH
        configs['split_txt_PATH'] = FLAGS.split_txt_PATH
        configs['model'] = ['art_painting', 'cartoon', 'photo', 'sketch']
        configs['split'] = ['train', 'test']
    elif data_source == 'mini-imagenet':
        configs['PATH'] = "/data2/hsq/mini-Imagenet"
        configs['split_txt_PATH'] = "/data2/hsq/mini-imagenet-split"
        configs['model'] = ['photo', 'sketch']
        configs['split'] = ['train', 'test', 'val']
    return configs.values()
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
def sample_imagenet(query_num_per_class_per_model=1, class_num=5,
                    support_num_per_class_per_model=1, train=True):

    raw_path, split_txt, model, train_test = config('mini-imagenet')
    if train:
        train_or_test = 'train'
    else:
        train_or_test = 'test'
    split_csv_path = [os.path.join(split_txt, t) for t in os.listdir(split_txt) if train_or_test in t][0]
    fliename_label = [readcsv(split_csv_path, os.path.join(raw_path, m)) for m in model]
    labels0 = set([fl['label'] for fl in fliename_label[1]])
    class_num = len(labels0)
    total_group = {m:[] for m in model}
    for k, m in enumerate(model):
        group = [[] for i in range(class_num)]
        for i, c in enumerate(labels0):
            group[i] = [fl['path'] for fl in fliename_label[k] if c == fl['label']]
        total_group[m] = group
    return total_group
a = sample_imagenet()
print(a['photo'][0][0])
print(a['sketch'][0][0])
