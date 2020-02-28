#-*-coding:utf-8-*-
from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import flags
from tensorflow.python.client import timeline
from data_generator import DataGenerator
import pandas as pd
from model import Model
import numpy as np
import utils
import tensorflow as tf
import utils
import math
from tqdm import tqdm
FLAGS = flags.FLAGS
##config dataset
# flags.DEFINE_string("data_PATH", default="/data2/hsq/Project/PACS/", help="The dataset's path.")
# flags.DEFINE_string("split_txt_PATH", default="/data2/hsq/Project/multiModelMetric/pacs_filename", help="file with how to split row data.")

flags.DEFINE_string("data_PATH", default="/data2/hsq/mini-Imagenet", help="The dataset's path.")
flags.DEFINE_string("split_txt_PATH", default="/data2/hsq/Project/mini-imagenet-split", help="file with how to split row data.")
flags.DEFINE_string("meta_data_path", default="/data2/hsq/Project/mini-imagenet-tasks-data", help="npy file path.")
flags.DEFINE_string("data_source", default="mini-imagenet", help="The dataset's name.")
flags.DEFINE_bool("visualize", default=False, help="visualize or not.")


flags.DEFINE_integer("image_size", default=84, help="input image channels.")
flags.DEFINE_integer("model", default=2, help="The num of data model.")
# flags.DEFINE_integer("num_class", default=7, help="The num of category.")

##config model
flags.DEFINE_integer("k_neighbor", default=1, help="the number of k-nearest neighbors.")
flags.DEFINE_integer("input_dim", default=3, help="input image channels.")
flags.DEFINE_string("backbone", default="Conv64F", help="Model name.")
flags.DEFINE_integer("filter_num", default=64, help="Model name.")
flags.DEFINE_string("distance_style", default="euc_norm", help="how to compute the distance.")
flags.DEFINE_bool("max_pool", default=True, help="use maxpool or not.")
flags.DEFINE_string("norm", default="None", help="choose norm style.")
flags.DEFINE_float("margin", default=1.0, help="set the margin of the loss_eps.")
flags.DEFINE_float("loss_weight", default=0.5, help="set the weight of the loss.")
flags.DEFINE_bool("eps_usehard", default=False, help="eps use hard or not.")
flags.DEFINE_bool("eps_loss", default=False, help="eps use or not.")
flags.DEFINE_bool("category_loss", default=True, help="category loss use or not.")
flags.DEFINE_bool("same_class_dist", default=False, help="turn on class dist loss use or not.")
flags.DEFINE_string("init_style", default='normal', help="how to initialize weight parameters.")
# flags.DEFINE_bool("pop", default=True, help="conv pop or not.")
flags.DEFINE_bool("support_weight", default=False, help="use support weight or not.")
flags.DEFINE_bool("intra_var", default=False, help="use intra weight or not.")
flags.DEFINE_bool("inter_var", default=False, help="use inter var weight or not.")
flags.DEFINE_bool("prototype", default=True, help="use prototype or not.")
flags.DEFINE_string("optimizer", default='adam', help="how to optimize parameters.")
flags.DEFINE_bool("adapt_prototypes", default=False, help="how to optimize parameters.")
flags.DEFINE_integer("finetune_times", default=5, help="finetune times.")
flags.DEFINE_float("finetune_lr", default=0.01, help="finetune lr.")






##config train
flags.DEFINE_integer("task_num", default=300000, help="the total number of training episodes.")
flags.DEFINE_integer("episode_tr", default=10000, help="the total number of training episodes.")
flags.DEFINE_integer("episode_val", default=20, help="the total number of evaluate episodes.")
flags.DEFINE_integer("episode_ts", default=1000, help="the total number of testing episodes.")
flags.DEFINE_integer("test_batch_size", default=1, help="the test batch size.")
flags.DEFINE_bool("load_ckpt", default=False, help="load check point or not.")
flags.DEFINE_bool("save_ckpt", default=True, help="save check point or not.")

flags.DEFINE_bool("debug_mode", default=False, help="debug or not.")


flags.DEFINE_integer("support_num", default=3, help="Num of support per class per model.")
flags.DEFINE_integer("query_num", default=7, help="Num of query per class per model.")
flags.DEFINE_integer("way_num", default=5, help="the number of classify ways.")
flags.DEFINE_integer("pretrain_iteration", default=2, help="pre train iterations.")
flags.DEFINE_integer("iteration", default=30, help="iterations.")
flags.DEFINE_float("lr", default=0.001, help="learning rate.")
flags.DEFINE_bool("train", default=True, help="Train or not.")
flags.DEFINE_bool("lr_decay", default=True, help="lr_decay or not.")
flags.DEFINE_integer("decay_iteration", default=10, help="lr_decay or not.")
flags.DEFINE_float("decay_rate", default=0.05, help="learning rate decay rate.")
flags.DEFINE_string("model_path", default="/data2/hsq/Project/multiModelMetric/log/model_checkpoint/300000_3shot_7q_001_eucnorm_adapt", help="model's path.")
flags.DEFINE_string("loss_function", default="sce", help="choose loss function.")
flags.DEFINE_string("gpu", default='1', help="choose gpu.")

FLAGS = flags.FLAGS
import os
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
loss_line = {'train_loss': [], 'train_accu': [], 'test_accu': [], 'test_loss': []}


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


def test_iteration(sess, model, acc_loss, bestacc, test_tasks, i, saver):
    # saver = tf.train.Saver()
    if FLAGS.lr_decay and i % 10 == 0 and i != 0: model.decay(i)
    test_acc, test_loss = 0.0, 0.0
    print("\n================================testing================================\n")
    # b = FLAGS.test_batch_size

    for k in tqdm(range(int(FLAGS.episode_ts))):
        task = test_tasks[k]
        support_x = task['support_set'][0]
        support_y = task['support_set'][1]
        query_x = task['query_set'][0]
        query_y = task['query_set'][1]
        support_m = task['support_set'][2]

        feed_dic = {model.support_x: support_x, model.query_x: query_x,
                    model.support_y: support_y, model.query_y: query_y,
                    model.support_m: support_m}

        with sess.as_default():
            acc, loss = sess.run(acc_loss, feed_dic)

        test_acc += acc
        test_loss += loss
    ts_accurcy = test_acc / FLAGS.episode_ts
    ts_loss = test_loss / FLAGS.episode_ts
    print("\niter %d  test acc is %f, test loss is %f." % ((i * 10000), ts_accurcy, ts_loss))
    loss_line['test_accu'].append(ts_accurcy)
    loss_line['test_loss'].append(ts_loss)
    print("\nlearning rate is:", model.lr.eval())
    if (ts_accurcy > bestacc):
        bestacc = ts_accurcy
        if FLAGS.save_ckpt:
            if not os.path.exists(FLAGS.model_path):
                os.makedirs(FLAGS.model_path)
            saver.save(sess, FLAGS.model_path + '/model', global_step=i)
    return bestacc



def train(test_tasks):


    model = Model()
    sess = tf.InteractiveSession()
    model.construct_model()

    # First let's load meta graph and restore weights

        # with tf.Session() as sess_load:
        #


        #     graph = tf.get_default_graph()
        #
        #     names = [n.name for n in graph.as_graph_def().node]
        #     [print(n) for n in names if 'conv' in n]
        #     model.construct_model(graph)

    # if not FLAGS.load_ckpt:
    trainop, acc_loss = model.trainop()
    # _, model.trainop(train=False)
    tf.global_variables_initializer().run()

    if FLAGS.load_ckpt:
        model_meta_file = [os.path.join(FLAGS.model_path, m) for m in os.listdir(FLAGS.model_path) if '.meta' in m]
        print("model path is:", model_meta_file)
        saver = tf.compat.v1.train.import_meta_graph(model_meta_file[-1])
        print("================================loading model================================")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model_path))
    else:
        saver = tf.train.Saver()
    print("================================generating tasks================================")
    task_filenames = np.load("300000Tasks_5way_3shot_14query_miniimagenet.npy", allow_pickle=True)
    print("task nums is:", len(task_filenames))
    bestacc = 0.0
    bigbatch_size = int(len(task_filenames)/FLAGS.iteration)

    # if FLAGS.load_ckpt:
    #     ckpt = tf.train.get_checkpoint_state("/data2/hsq/Project/multiModelMetric/log/model_checkpoint/mini-imagenet_5way_1shot_5000tasks")
    #     print(ckpt.model_checkpoint_path)
    #     tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
    for i in (range(FLAGS.pretrain_iteration, FLAGS.iteration)):
        l = a = 0.0
        bigbatch = task_filenames[i * bigbatch_size:(i + 1) * bigbatch_size]
        print("\n================================epoch%d================================\n"%(i))
        for j in tqdm(range(bigbatch_size)):
            task = bigbatch[j]
            support_set, query_set = utils.make_set_tensor(task['support_set']), utils.make_set_tensor(task['query_set'])
            feed_dic = {model.support_x: support_set[0], model.query_x: query_set[0],
                        model.support_y: support_set[1], model.query_y: query_set[1],
                        model.support_m: support_set[2], model.query_m: query_set[2]}


            with sess.as_default():
                # options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # _, la = sess.run([trainop, acc_loss], feed_dict=feed_dic, options=options, run_metadata=run_metadata)
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # with open('timeline_01.json', 'w') as f:
                #     f.write(chrome_trace)

                _, al = sess.run([trainop, acc_loss], feed_dict=feed_dic)

            if(i == 0 and j == 0 and FLAGS.visualize):
                visualize(sess, graph=True)
            l += al[1]
            a += al[0]
            if (j+1) % 500 == 0:
                bestacc = test_iteration(sess, model, acc_loss, bestacc, test_tasks, i, saver)
                print("\niter %d train loss is %f, train acc is %f.\n" % (
                i*bigbatch_size + j, l / (j+1), a / (j+1)))



        del bigbatch

        print("\nepoch %d train loss is %f, train acc is %f.\n"%(i,l/FLAGS.episode_tr, a/FLAGS.episode_tr))
        loss_line['train_loss'].append(l/FLAGS.episode_tr)
        loss_line['train_accu'].append(a/FLAGS.episode_tr)



    np.savetxt(os.path.join(FLAGS.model_path, 'loss_acc.csv'), np.array(loss_line))
    pd.DataFrame(loss_line).to_csv(os.path.join(FLAGS.model_path, 'loss_acc.csv'))


import multiprocessing
def get_test_task(test_generator):
    test_file = test_generator.generator_tasks()
    all_task = []
    for task in tqdm(test_file):
        support_set, query_set = utils.make_set_tensor(task['support_set']), utils.make_set_tensor(task['query_set'])
        all_task.append({'support_set': support_set, 'query_set': query_set})
    # task_support_x = np.array([task['support_set'][0] for task in all_task]).astype(np.float)
    # task_support_y = np.array([task['support_set'][1] for task in all_task]).astype(np.float)
    # task_query_x = np.array([task['query_set'][0] for task in all_task]).astype(np.float)
    # task_query_y = np.array([task['query_set'][1] for task in all_task]).astype(np.float)
    # task_support_m = np.array([task['support_set'][2] for task in all_task]).astype(np.float)
    # task_query_m = np.array([task['query_set'][2] for task in all_task]).astype(np.float)
    # return [task_support_x, task_support_y, task_query_x, task_query_y, task_support_m, task_query_m]
    return all_task
def main():
    data_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num, FLAGS.train)
    test_generator = DataGenerator(FLAGS.query_num, FLAGS.support_num, train=False)
    data_name = 'test' + str(FLAGS.episode_ts)+'tasks_'+str(FLAGS.query_num)+'q_'+str(FLAGS.image_size)+'_'+str(FLAGS.support_num)+'shot'
    tasks_path = [p for p in os.listdir(FLAGS.meta_data_path) if data_name in p]
    if len(os.listdir(FLAGS.meta_data_path)) == 0 or len(tasks_path) == 0:
        all_task = get_test_task(test_generator)
        if FLAGS.episode_tr > 500:
            for i in range(math.ceil(len(all_task)/500)):
                tasks = all_task[i*500:(i+1)*500]
                np.save(FLAGS.meta_data_path+'/'+data_name + 'part'+str(i)+ '.npy', tasks, allow_pickle=True)
        else:
            np.save(FLAGS.meta_data_path + '/' + data_name + '.npy', all_task, allow_pickle=True)
    else:
        print(tasks_path)
        all_task=[]
        for t in tasks_path:
            all_task.extend(np.load(os.path.join(FLAGS.meta_data_path, t), allow_pickle=True))
        # task_support_x = np.array([task['support_set'][0] for task in all_task]).astype(np.float)
        # task_support_y = np.array([task['support_set'][1] for task in all_task]).astype(np.float)
        # task_query_x = np.array([task['query_set'][0] for task in all_task]).astype(np.float)
        # task_query_y = np.array([task['query_set'][1] for task in all_task]).astype(np.float)
        # task_support_m = np.array([task['support_set'][2] for task in all_task]).astype(np.float)
        # task_query_m = np.array([task['query_set'][2] for task in all_task]).astype(np.float)
        # all_task = [task_support_x, task_support_y, task_query_x, task_query_y, task_support_m, task_query_m]
        print(len(all_task))

    test_tasks = all_task

    if FLAGS.train :
        train(test_tasks)
    # else:
    #     test(model, data_generator)
    exit(0)


    pass
if __name__ == '__main__':
    main()
