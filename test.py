#-*-coding:utf-8-*-
import tensorflow as tf
qy = [1,0,0]
sy = [[0,0,1],[1,0,0],[0,1,0],[1,0,0]]
qx = [1.0, 2.0, 1.0]
sx = [[2.0, 2.0, 2.0], [1.0, 1.0, 2.0], [2.0, 0.0, 1.0], [0.0, 3.0, 2.0]]
qy = tf.cast(tf.convert_to_tensor(qy), dtype=tf.float32)
sy = tf.cast(tf.convert_to_tensor(sy), dtype=tf.float32)
qx = tf.cast(tf.convert_to_tensor(qx), dtype=tf.float32)
sx = tf.cast(tf.convert_to_tensor(sx), dtype=tf.float32)

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

dist = distance(qx, sx)

with tf.Session() as sess:
    differ_matrix = get_differ_matrix(qy, sy)
    print(qy.eval())
    print(sy.eval())
    print(differ_matrix.eval())
    print(get_differ_matrix(qy, sy[1]).eval())