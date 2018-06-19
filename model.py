import os
import numpy as np
import random
import scipy.misc as scm
import tensorflow as tf

from collections import namedtuple
from tqdm import tqdm
from glob import glob
from tensorflow.examples.tutorials.mnist import input_data

import module
# import util 

class Network(object):
    
    def __init__(self, sess, args):
        self.sess = sess
        self.phase = args.phase
        self.continue_train = args.continue_train
        self.data_dir = args.data_dir
        self.log_dir = args.log_dir
        self.ckpt_dir = args.ckpt_dir
        self.sample_dir = args.sample_dir
        self.test_dir = args.test_dir
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.input_size = args.input_size
        self.image_c = args.image_c
        self.label_n = args.label_n
        self.nf = args.nf
        self.lr = args.lr
        self.beta1 = args.beta1
        self.sample_step = args.sample_step
        self.log_step = args.log_step
        self.ckpt_step = args.ckpt_step
        
        # hyper parameter for building module
        OPTIONS = namedtuple('options', ['batch_size', 'nf', 'label_n', 'phase'])
        self.options = OPTIONS(self.batch_size, self.nf, self.label_n, self.phase)
        
        # build model & make checkpoint saver
        self.build_model()
        self.saver = tf.train.Saver()
        
    
    def build_model(self):
        # placeholder
        self.input_images = tf.placeholder(tf.float32, 
                                          [None,self.input_size,self.input_size,self.image_c],
                                          name='input_images')
        self.labels = tf.placeholder(tf.float32, [None,self.label_n], name='labels')
        
        # loss funciton
        # self.pred = module.classifier(self.input_images, self.options, reuse=False, name='densenet')
        self.pred = module.DenseNet(self.input_images, self.nf, self.label_n, self.phase).model
        self.loss = module.cls_loss(logits=self.pred, labels=self.labels)
        
        # accuracy
        corr = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))    
        self.accr_count = tf.reduce_sum(tf.cast(corr, "float"))

        # trainable variables
        t_vars = tf.trainable_variables()
#         self.module_vars = [var for var in t_vars if 'densenet' in var.name]
#         for var in t_vars: print(var.name)
        
        # optimizer
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=t_vars)
        
        # placeholder for summary
        self.total_loss = tf.placeholder(tf.float32)
        self.accr = tf.placeholder(tf.float32)
        
        # summary setting
        self.summary()
        
    def train(self):
        
        # load train data list
        mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
        batch_idxs = mnist.train.num_examples // self.batch_size
        self.train_images, self.train_labels = mnist.train.images, mnist.train.labels
        self.train_images = self.train_images.reshape([-1, self.input_size-4, self.input_size-4, self.image_c])
        self.train_images = self.zero_padding(self.train_images) # 28*28*1 -> 32*32*1
        self.test_images, self.test_labels = mnist.test.images, mnist.test.labels
        self.test_images = self.test_images.reshape([-1, self.input_size-4, self.input_size-4, self.image_c])
        self.test_images = self.zero_padding(self.test_images) # 28*28*1 -> 32*32*1
        
        # variable initialize
        self.sess.run(tf.global_variables_initializer())
            
        # load or not checkpoint
        if self.continue_train and self.checkpoint_load():
            print(" [*] before training, Load SUCCESS ")
        else:
            print(" [!] before training, no need to Load ")    
        
        count_idx = 0
        # train
        for epoch in range(self.epoch):
            print('Epoch[{}/{}]'.format(epoch+1, self.epoch))
            cost = 0
            
            for i in tqdm(range(batch_idxs)):
                # get batch images and labels
                images, labels = mnist.train.next_batch(self.batch_size)
                images = images.reshape([self.batch_size, self.input_size-4, self.input_size-4, self.image_c])
                images = self.zero_padding(images) # 28*28*1 -> 32*32*1
                          
                # update network
                feeds = {self.input_images: images, self.labels: labels}
                _, summary_loss = self.sess.run([self.optim, self.sum_loss], feed_dict=feeds)
              
                count_idx += 1
                
                # log step (summary)
                if count_idx % self.log_step == 0:
                    train_accr = self.accuracy('train')
                    valid_accr = self.accuracy('test')
                    
                    self.writer_cost.add_summary(summary_loss, count_idx)

                    summary = self.sess.run(self.sum_accr, feed_dict={self.accr:train_accr})
                    self.writer_train_accr.add_summary(summary, count_idx)

                    summary = self.sess.run(self.sum_accr, feed_dict={self.accr:valid_accr})
                    self.writer_valid_accr.add_summary(summary, count_idx)
                
                # checkpoint step
                if count_idx % self.ckpt_step == 0:
                    self.checkpoint_save(count_idx)


    def test(self):
        # load test data
        mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
        self.test_images, self.test_labels = mnist.test.images, mnist.test.labels
        self.test_images = self.test_images.reshape([-1, self.input_size-4, self.input_size-4, self.image_c])
        self.test_images = self.zero_padding(self.test_images) # 28*28*1 -> 32*32*1
        
        ## test for batch_normalization & drop-out train/test phase!!!!
        perm = np.random.permutation(self.test_images.shape[0])
        self.test_images = np.take(self.test_images, perm, axis=0)
        self.test_labels = np.take(self.test_labels, perm, axis=0)
        
        self.sess.run(tf.global_variables_initializer())
        
        # load checkpoint
        if self.checkpoint_load():
            print(" [*] checkpoint load SUCCESS ")
        else:
            print(" [!] checkpoint load failed ")
        
        # print test accuracy
        self.accuracy('test')
        
   
    
    def summary(self):
        # summary writer
        self.writer_cost = tf.summary.FileWriter(os.path.join(self.log_dir,'cost'), self.sess.graph)
        self.writer_train_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'train_accr'),self.sess.graph)
        self.writer_valid_accr = tf.summary.FileWriter(os.path.join(self.log_dir,'valid_accr'),self.sess.graph)        
        
        # summary session
        self.sum_loss = tf.summary.scalar('loss value',self.loss)
        self.sum_accr = tf.summary.scalar('train_accr', self.accr)


    def load_data(self):
        self.X_train, self.y_train = util.load_mnist(self.dataset_dir, kind='train')
        self.n_train = np.size(self.X_train, 0)
        
        self.X_test, self.y_test = util.load_mnist(self.dataset_dir, kind='t10k')
        self.n_test = np.size(self.X_test, 0)


    def accuracy(self, phase='test'):
        # train or test or validate
        Dataset = namedtuple('Dataset',['X_', 'y_', 'n_'])
        if phase=='train':
            dataset = Dataset(self.train_images, self.train_labels, 55000)
        elif phase=='test':
            dataset = Dataset(self.test_images, self.test_labels, 10000)
    
        # accuracy
        accr = 0.    
        for i in range(dataset.n_ // self.batch_size):
            feeds = {
                    self.input_images: dataset.X_[i*self.batch_size : (i+1)*self.batch_size],
                    self.labels: dataset.y_[i*self.batch_size : (i+1)*self.batch_size]
                    }
            accr += self.sess.run(self.accr_count, feed_dict=feeds)
        accr = accr / dataset.n_
        
        print('{} accuracy: {:.04f}'.format(phase, accr))
        return accr 


    def checkpoint_save(self, count):
        model_name = "net.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name),
                        global_step=count)
    
    
    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            return True
        else:
            return False
        
    def zero_padding(self, images):
        pad_imgs = np.zeros([images.shape[0], self.input_size, self.input_size, self.image_c]) # 32*32
        pad_imgs[:,2:-2,2:-2,:] = images
        return pad_imgs # 32*32*1