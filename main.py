#-*-coding:utf-8-*-
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import flags
from data_generator import DataGenerator
from model import Model
import numpy as np
import tensorflow as tf
import utils
from tqdm import tqdm
FLAGS = flags.FLAGS
##config dataset
flags.DEFINE_string("data_PATH", default="/home/nnu/hsq/PACS/", help="The dataset's path.")
flags.DEFINE_string("split_txt_PATH", default="/home/nnu/hsq/metric_PACS/pacs_filename", help="file with how to split row data.")
flags.DEFINE_string("data_source", default="PACS", help="The dataset's name.")
flags.DEFINE_integer("image_size", default=64, help="input image channels.")
flags.DEFINE_integer("model", default=4, help="The num of data model.")
flags.DEFINE_integer("num_class", default=7, help="The num of category.")

##config model
flags.DEFINE_integer("k_neighbor", default=1, help="the number of k-nearest neighbors.")
flags.DEFINE_integer("input_dim", default=3, help="input image channels.")
flags.DEFINE_string("backbone", default="Conv", help="Model name.")
flags.DEFINE_integer("filter_num", default=64, help="Model name.")
flags.DEFINE_string("distance_style", default="euc", help="how to compute the distance.")
flags.DEFINE_bool("maxpool", default=True, help="use maxpool or not.")
flags.DEFINE_string("norm", default="None", help="choose norm style.")
flags.DEFINE_bool("max_pool", default=True, help="use maxpool or not")



##config train
flags.DEFINE_integer("episode_tr", default=2000, help="the total number of training episodes.")
flags.DEFINE_integer("episode_val", default=1000, help="the total number of evaluate episodes.")
flags.DEFINE_integer("episode_ts", default=100, help="the total number of testing episodes.")
flags.DEFINE_integer("support_num", default=1, help="Num of support per class per model.")
flags.DEFINE_integer("query_num", default=1, help="Num of query per class per model.")
flags.DEFINE_integer("way_num", default=5, help="the number of classify ways.")
flags.DEFINE_integer("iteration", default=10000, help="iterations.")
flags.DEFINE_float("lr", default=0.001, help="learning rate.")
flags.DEFINE_bool("train", default=True, help="Train or not.")
flags.DEFINE_bool("lr_decay", default=True, help="lr_decay or not.")
flags.DEFINE_bool("visualize", default=True, help="visualize or not.")
flags.DEFINE_float("decay_rate", default=0.999, help="learning rate decay rate.")
flags.DEFINE_string("model_path", default="log/model_checkpoint/metric_s1_q1_checkpoint", help="model's path.")
flags.DEFINE_string("loss_function", default="mse", help="choose loss function.")
FLAGS = flags.FLAGS

def visualize(sess, graph=False):
    if graph:
        writer = tf.summary.FileWriter("log/", sess.graph)
    # tf.global_variables_initializer().run()
    writer.close()
def make_test_tast(test_tasks):
    task_support_x = np.array([t['support_set'][0] for t in test_tasks], dtype=np.float)
    task_support_y = np.array([t['support_set'][1] for t in test_tasks], dtype=np.float)
    task_query_x = np.array([t['query_set'][0] for t in test_tasks], dtype=np.float)
    task_query_y = np.array([t['query_set'][1] for t in test_tasks], dtype=np.float)
    return tf.convert_to_tensor(task_support_x, dtype=tf.float32), \
           tf.convert_to_tensor(task_support_y, dtype=tf.float32), tf.convert_to_tensor(task_query_x, dtype=tf.float32), tf.convert_to_tensor(task_query_y, dtype=tf.float32)

def test_iteration(sess, model, task_support_x, task_support_y, task_query_x, task_query_y):
    num = task_query_y.shape[0]
    # feed_dic = {model.support_x: task_support_x, model.query_x: task_query_x,
    #                     model.support_y: task_support_y, model.query_y: task_query_y}
    losses = tf.map_fn(fn=lambda inp: model.get_loss(inp, model.weights), elems=(task_support_x, task_support_y, task_query_x, task_query_y), dtype=tf.float32)
    with sess.as_default():
        loss, acc = sess.run(losses)
    print("test loss is %f, acc is %f."%(sum(loss)/num, sum(acc)/num))


def train(model, data_generator, test_tasks):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    # task_support_x, task_support_y, task_query_x, task_query_y = make_test_tast(test_tasks)
    all_task = data_generator.make_data_tensor()
    trainop, acc_loss = model.trainop()
    tf.global_variables_initializer().run()
    print("training...")
    bestacc = 0.0
    for i in range(FLAGS.iteration):
        l, a = 0, 0
        for j in tqdm(range(FLAGS.episode_tr)):
            task = all_task[j]
            feed_dic = {model.support_x: task['support_set'][0], model.query_x: task['query_set'][0],
                        model.support_y: task['support_set'][1], model.query_y: task['query_set'][1]}
            with sess.as_default():
                _, la = sess.run([trainop, acc_loss], feed_dic)
                # output_s = model.forward(model.support_x, model.weights, reuse=True)
                # output_q = model.forward(model.query_x, model.weights, reuse=True)

                # comloss = sess.run(utils.compute_loss((output_q, model.query_y), output_s, model.support_y), feed_dic)

            if(i == 0 and j == 0 and FLAGS.visualize):
                visualize(sess, graph=True)
            l += la[0]
            a += la[1]
        if i % 10 == 0:
            if FLAGS.lr_decay: model.decay()
            test_acc, tl = 0.0, 0.0
            print("testing...")
            for j in tqdm(range(FLAGS.episode_ts)):
                task = test_tasks[j]
                feed_dic = {model.support_x: task['support_set'][0], model.query_x: task['query_set'][0],
                            model.support_y: task['support_set'][1], model.query_y: task['query_set'][1]}
                with sess.as_default():
                    test_loss, acc = sess.run(model.get_loss((model.support_x, model.support_y, model.query_x, model.query_y), model.weights,), feed_dic)
                test_acc += acc
                tl += sum(test_loss)
            print("\n epoch %d train loss is %f, train acc is %f." % (i, tl / FLAGS.episode_ts, test_acc / FLAGS.episode_ts))
        print("\n learning rate is:", model.lr.eval())
        print("\n epoch %d train loss is %f, train acc is %f."%(i,l/FLAGS.episode_tr, a/FLAGS.episode_tr))
        # test_iteration(sess, model, task_support_x, task_support_y, task_query_x, task_query_y)
        if (i > 100 and (i % 100 == 0 or a/FLAGS.episode_tr >= bestacc)):
            saver.save(sess, FLAGS.model_path, global_step=i)



def test(model, data_generator):
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    all_test_task = data_generator.make_data_tensor()
    l, a = 0, 0
    for j in tqdm(range(FLAGS.episode)):
        task = all_test_task[j]
        feed_dic = {model.support_x: task['support_set'][0], model.query_x: task['query_set'][0],
                    model.support_y: task['support_set'][1], model.query_y: task['query_set'][1]}
        with sess.as_default():
            saver.restore(sess, FLAGS.model_path)
            la = sess.run(model.get_loss(task, model.weights), feed_dic)
            l += la[0]
            a += la[1]
    print("\n test loss is %f, acc is %f" % (l / FLAGS.episode, a / FLAGS.episode))


def main():
    data_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num, FLAGS.train)
    test_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num, train=False)
    test_tasks = test_generator.make_data_tensor()
    model = Model()
    model.construct_model()
    if FLAGS.train :
        train(model, data_generator, test_tasks)
    else:
        test(model, data_generator)
    exit(0)


    pass
if __name__ == '__main__':
    main()
