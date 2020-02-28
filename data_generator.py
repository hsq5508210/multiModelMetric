#-*-coding:utf-8-*-
import numpy as np
import os
import random
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python.platform import flags
from utils import sample_task
from utils import make_set_tensor
import utils

FLAGS = flags.FLAGS

class DataGenerator(object):
    def __init__(self, query_num_per_class_per_model, support_num_per_class_per_model, train):
        self.query_num_per_class_per_model = query_num_per_class_per_model
        self.support_num_per_class_per_model = support_num_per_class_per_model
        raw_data_dir = FLAGS.data_PATH
        self.train = train
        if self.train:
            self.episode = FLAGS.episode_tr
        else:
            self.episode = FLAGS.episode_ts
        # if FLAGS.data_source == 'PACS':
        # self.num_class = FLAGS.num_class
        self.img_size = FLAGS.image_size
        self.input_dim = np.prod(self.img_size)*3
        # self.output_dim = self.num_class



    def generator_tasks(self,):
        """
        :param train: train or not.
        :return: all tasks, composed by dicts e.g.{'support_set': support_set, 'query_set':query_set}
                            which value in dicts is list of data tensors, first element such as support_set[0] is
                            image tensor, the next is label one-hot tensors.
        """
        print('===========================Generating task filedirs===========================')
        all_tasks = []
        # FLAGS.episode_tr
        if FLAGS.data_source == 'PACS':
            raw_path, split_txt, model, train_test = utils.config('PACS')
            data_model = utils.split(model, split_txt, self.train, raw_path)
            raw_class_num = 7
        elif FLAGS.data_source == 'mini-imagenet':
            raw_path, split_txt, model, train_test = utils.config('mini-imagenet')
            data_model, raw_class_num = utils.split_imagenet(model, split_txt, self.train, raw_path)
        if self.train:
            tasknum = FLAGS.task_num
        else:
            tasknum = FLAGS.episode_ts
        for _ in tqdm(range(tasknum)):
            task = utils.sample_task(data_model, raw_class_num, model, query_num_per_class_per_model=FLAGS.query_num,
                                     class_num=FLAGS.way_num,
                                     support_num_per_class_per_model=FLAGS.support_num, train=self.train)
            all_tasks.append({'support_set': task['support'], 'query_set': task['query']})
        return all_tasks




