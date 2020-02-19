#-*-coding:utf-8-*-
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import flags
from data_generator import DataGenerator
import pandas as pd
from model import Model
import numpy as np
import tensorflow as tf
import utils
from tqdm import tqdm
FLAGS = flags.FLAGS
##config dataset
# flags.DEFINE_string("data_PATH", default="/data2/hsq/Project/PACS/", help="The dataset's path.")
# flags.DEFINE_string("split_txt_PATH", default="/data2/hsq/Project/multiModelMetric/pacs_filename", help="file with how to split row data.")

flags.DEFINE_string("data_PATH", default="/data2/hsq/mini-Imagenet", help="The dataset's path.")
flags.DEFINE_string("split_txt_PATH", default="/data2/hsq/mini-imagenet-split", help="file with how to split row data.")
flags.DEFINE_string("meta_data_path", default="/data2/hsq/Project/mini-imagenet-tasks-data", help="npy file path.")
# flags.DEFINE_string("data_source", default="PACS", help="The dataset's name.")

flags.DEFINE_string("data_source", default="mini-imagenet", help="The dataset's name.")


flags.DEFINE_integer("image_size", default=64, help="input image channels.")
flags.DEFINE_integer("model", default=2, help="The num of data model.")
# flags.DEFINE_integer("num_class", default=7, help="The num of category.")

##config model
flags.DEFINE_integer("k_neighbor", default=1, help="the number of k-nearest neighbors.")
flags.DEFINE_integer("input_dim", default=3, help="input image channels.")
flags.DEFINE_string("backbone", default="Conv", help="Model name.")
flags.DEFINE_integer("filter_num", default=64, help="Model name.")
flags.DEFINE_string("distance_style", default="euc_v1", help="how to compute the distance.")
flags.DEFINE_bool("max_pool", default=True, help="use maxpool or not.")
flags.DEFINE_string("norm", default="None", help="choose norm style.")
flags.DEFINE_float("margin", default=1.0, help="set the margin of the loss_eps.")
flags.DEFINE_float("loss_weight", default=0.5, help="set the weight of the loss.")
flags.DEFINE_bool("eps_usehard", default=False, help="eps use hard or not.")
flags.DEFINE_bool("eps_loss", default=True, help="eps use or not.")
flags.DEFINE_bool("category_loss", default=True, help="category loss use or not.")
flags.DEFINE_bool("same_class_dist", default=False, help="turn on class dist loss use or not.")
flags.DEFINE_string("init_style", default='normal', help="how to initialize weight parameters.")
# flags.DEFINE_bool("pop", default=True, help="conv pop or not.")
flags.DEFINE_bool("support_weight", default=False, help="use support weight or not.")
flags.DEFINE_bool("intra_var", default=False, help="use intra weight or not.")
flags.DEFINE_bool("inter_var", default=False, help="use inter var weight or not.")









##config train
flags.DEFINE_integer("episode_tr", default=40, help="the total number of training episodes.")
flags.DEFINE_integer("episode_val", default=50, help="the total number of evaluate episodes.")
flags.DEFINE_integer("episode_ts", default=20, help="the total number of testing episodes.")
flags.DEFINE_bool("load_ckpt", default=False, help="load check point or not.")
flags.DEFINE_bool("save_ckpt", default=True, help="save check point or not.")

flags.DEFINE_bool("debug_mode", default=False, help="debug or not.")


flags.DEFINE_integer("test_batch_size", default=100, help="the test batch size.")
flags.DEFINE_integer("support_num", default=1, help="Num of support per class per model.")
flags.DEFINE_integer("query_num", default=1, help="Num of query per class per model.")
flags.DEFINE_integer("way_num", default=5, help="the number of classify ways.")
flags.DEFINE_integer("iteration", default=10000, help="iterations.")
flags.DEFINE_float("lr", default=0.0001, help="learning rate.")
flags.DEFINE_bool("train", default=True, help="Train or not.")
flags.DEFINE_bool("lr_decay", default=True, help="lr_decay or not.")
flags.DEFINE_integer("decay_iteration", default=5, help="lr_decay or not.")
flags.DEFINE_bool("visualize", default=False, help="visualize or not.")
flags.DEFINE_float("decay_rate", default=0.05, help="learning rate decay rate.")
flags.DEFINE_string("model_path", default="/data2/hsq/Project/multiModelMetric/log/model_checkpoint/mini-imagenet_5way_1shot_5000task_lossep_margin0.4_w0.3_euc/5_1_5000_losseps_margin0.4_w0.3_euc", help="model's path.")
flags.DEFINE_string("loss_function", default="log", help="choose loss function.")
FLAGS = flags.FLAGS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
loss_line = {'train_loss': [], 'train_accu': [], 'test_accu': []}


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

def test_iteration(sess, model, bestacc, test_tasks, i, j):
    saver = tf.train.Saver()
    if FLAGS.lr_decay and i % 5 == 0 and i != 0: model.decay(i)
    test_acc, tl = 0.0, 0.0
    print("testing...")
    task_support_x = np.array([task['support_set'][0] for task in test_tasks]).astype(np.float)
    task_support_y = np.array([task['support_set'][1] for task in test_tasks]).astype(np.float)
    task_query_x = np.array([task['query_set'][0] for task in test_tasks]).astype(np.float)
    task_query_y = np.array([task['query_set'][1] for task in test_tasks]).astype(np.float)
    b = FLAGS.test_batch_size
    for k in tqdm(range(int(FLAGS.episode_ts / FLAGS.test_batch_size))):
        support_x = task_support_x[k * b: (k + 1) * b]
        support_y = task_support_y[k * b: (k + 1) * b]
        query_x = task_query_x[k * b: (k + 1) * b]
        query_y = task_query_y[k * b: (k + 1) * b]
        feed_dic = {model.support_x: support_x, model.query_x: query_x,
                    model.support_y: support_y, model.query_y: query_y}
        with sess.as_default():
            # test_loss, acc = sess.run(model.get_loss((model.support_x, model.support_y, model.query_x, model.query_y), model.weights,), feed_dic)
            lr, acc = sess.run([model.lr, model.testop((support_x, support_y, query_x, query_y))], feed_dic)
        # acc = model.testop((support_x, support_y, query_x, query_y)).eval()
        test_acc += sum(acc)
        # tl += test_loss
    ts_accurcy = test_acc / FLAGS.episode_ts
    print("\nepoch %d  test acc is %f." % ((i + 1), ts_accurcy))
    loss_line['test_accu'].append(ts_accurcy)
    print("\nlearning rate is:", lr)
    if (ts_accurcy > bestacc):
        bestacc = ts_accurcy
        if FLAGS.save_ckpt:
            if not os.path.exists(FLAGS.model_path):
                os.makedirs(FLAGS.model_path)
            saver.save(sess, FLAGS.model_path + '/model', global_step=i)
    return bestacc



def train(model, data_generator, test_tasks):
    sess = tf.InteractiveSession()
    tasks_path = [p for p in os.listdir(FLAGS.meta_data_path) if str(FLAGS.episode_tr)+'tasks_'+str(FLAGS.image_size)+'_'+str(FLAGS.support_num)+'shot' in p]
    if len(os.listdir(FLAGS.meta_data_path)) == 0 or len(tasks_path) == 0:
        all_task = data_generator.make_data_tensor()
        np.save(FLAGS.meta_data_path+'/'+str(FLAGS.episode_tr)+'tasks_'+str(FLAGS.image_size)+'_'+str(FLAGS.support_num)+'shot' +'.npy', all_task, allow_pickle=True)
    else:
        print(tasks_path[0])
        all_task = np.load(os.path.join(FLAGS.meta_data_path, tasks_path[0]), allow_pickle=True)
    trainop, acc_loss = model.trainop()
    tf.global_variables_initializer().run()
    print("training...")
    bestacc = 0.0
    if FLAGS.load_ckpt:
        ckpt = tf.train.get_checkpoint_state("/data2/hsq/Project/multiModelMetric/log/model_checkpoint/mini-imagenet_5way_1shot_5000tasks")
        print(ckpt.model_checkpoint_path)
        tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    for i in range(FLAGS.iteration):
        l, a = 0, 0
        for j in tqdm(range(FLAGS.episode_tr)):
            task = all_task[j]
            feed_dic = {model.support_x: task['support_set'][0], model.query_x: task['query_set'][0],
                        model.support_y: task['support_set'][1], model.query_y: task['query_set'][1],
                        model.support_m: task['support_set'][2],  model.query_m: task['query_set'][2]}
            with sess.as_default():
                _, la = sess.run([trainop, acc_loss], feed_dic)
                # output_s = model.forward(model.support_x, model.weights, reuse=True)
                # output_q = model.forward(model.query_x, model.weights, reuse=True)

                # comloss = sess.run(utils.compute_loss((output_q, model.query_y), output_s, model.support_y), feed_dic)
            if(i == 0 and j == 0 and FLAGS.visualize):
                visualize(sess, graph=True)
            l += la[0]
            a += la[1]
            # if ((j+1)) % 5000 == 0:
        bestacc = test_iteration(sess, model, bestacc, test_tasks, i, j)
        print("\nepoch %d train loss is %f, train acc is %f.\n"%(i+1,l/FLAGS.episode_tr, a/FLAGS.episode_tr))
        loss_line['train_loss'].append(l/FLAGS.episode_tr)
        loss_line['train_accu'].append(a/FLAGS.episode_tr)
        if FLAGS.debug_mode:
            task = all_task[0]
            feed_dic = {model.support_x: task['support_set'][0], model.query_x: task['query_set'][0],
                        model.support_y: task['support_set'][1], model.query_y: task['query_set'][1],
                        model.support_m: task['support_set'][2],  model.query_m: task['query_set'][2]}
            # print(task['support_set'][0])
            if i == 0:
                with sess.as_default():
                    output_s = sess.run(model.forward(model.support_x, model.weights, reuse=True), feed_dic)
                    output_q = sess.run(model.forward(model.query_x, model.weights, reuse=True), feed_dic)
                    # task_losses, losses_eps = dist1 = sess.run(model.debuf_nan(output_q, output_s), feed_dic)
                    losses = sess.run(model.debuf_nan(output_q, output_s), feed_dic)
            else:
                with sess.as_default():
                    output_s = sess.run(model.forward(model.support_x, model.weights, reuse=True), feed_dic)
                    output_q = sess.run(model.forward(model.query_x, model.weights, reuse=True), feed_dic)
                    # task_losses, losses_eps = sess.run(model.debuf_nan(output_q, output_s), feed_dic)
                    losses = sess.run(model.debuf_nan(output_q, output_s), feed_dic)

                    weight = sess.run(model.weights['conv1'], feed_dic)
                    # print(weight)
            print(losses)

            # print(dist1-dist0)



        if np.isnan(l):
            task = all_task[0]
            feed_dic = {model.support_x: task['support_set'][0], model.query_x: task['query_set'][0],
                        model.support_y: task['support_set'][1], model.query_y: task['query_set'][1],
                        model.support_m: task['support_set'][2],  model.query_m: task['query_set'][2]}

            with sess.as_default():
                output_s = sess.run(model.forward(model.support_x, model.weights, reuse=True), feed_dic)
                output_q = sess.run(model.forward(model.query_x, model.weights, reuse=True), feed_dic)
                gvs = sess.run(model.debuf_nan(output_q, output_s), feed_dic)
                weight = sess.run(model.weights['conv1'], feed_dic)
                # print(weight)
                # print(output_s.shape)
                # print(output_q.shape)
                #
                # # le = sess.run([model.predict_category()], feed_dic)
                # print([grad for grad, var in gvs])
            # np.savetxt(os.path.join(FLAGS.model_path, 'loss_acc.csv'), np.array(loss_line))
            pd.DataFrame(loss_line).to_csv(os.path.join(FLAGS.model_path, 'loss_acc.csv'))
            break
        # test_iteration(sess, model, task_support_x, task_support_y, task_query_x, task_query_y)

    # np.savetxt(os.path.join(FLAGS.model_path, 'loss_acc.csv'), np.array(loss_line))
    pd.DataFrame(loss_line).to_csv(os.path.join(FLAGS.model_path, 'loss_acc.csv'))


import multiprocessing
def main():
    data_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num, FLAGS.train)
    test_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num, train=False)
    test_tasks = test_generator.make_data_tensor()
    # print(test_tasks)
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
