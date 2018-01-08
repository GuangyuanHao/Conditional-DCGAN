from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from module import *
from utils import *
# model train test
class dcgan(object):
    def __init__(self, sess, args):
        self.sess= sess
        self.dataset_dir = args.dataset_dir
        self.batch_size = args.batch_size
        self.gf_dim = args.ngf
        self.df_dim = args.ndf
        self.z_dim = args.z_dim
        self.output_c_dim = args.output_nc
        self.input_c_dim = args.input_nc
        self.image_size = args.fine_size
        self.cg = cgenerator
        self.cd = cdiscriminator
        self.lsgan = mae
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                    args.ngf, args.ndf, args.output_nc))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):

        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim],
                                        name="real_AB")
        self.real_img = self.real_data
        self.y = tf.placeholder(tf.float32,[None,10],name="label_y")
        self.z = tf.placeholder(tf.float32,[None, self.z_dim], name="noise_z")

        self.fake_img = self.cg(self.z,self.y,self.options, reuse = False, name="cg" )
        self.dfake_img = self.cd(self.fake_img,self.y,self.options,  reuse= False, name="cd")
        self.dreal_img = self.cd(self.real_img, self.y,self.options, reuse = True, name="cd")

        self.g_loss = self.lsgan(self.dfake_img, tf.ones_like(self.dfake_img))
        self.d_loss_real = self.lsgan(self.dreal_img,tf.ones_like(self.dreal_img))
        self.d_loss_fake = self.lsgan(self.dfake_img,tf.zeros_like(self.dfake_img))
        self.d_loss= (self.d_loss_real + self.d_loss_fake)

        self.g_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_sum = tf.summary.scalar("db_loss", self.d_loss)
        self.d_loss_real_sum = tf.summary.scalar("db_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.d_loss_fake)

        self.d_sum = tf.summary.merge(
            [self.d_sum,self.d_loss_real_sum,self.d_loss_fake_sum]
        )

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if "cd" in var.name]
        self.g_vars = [var for var in t_vars if "cg" in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):

        self.d_optim = tf.train.AdamOptimizer(args.lr, beta1= args.beta1).\
            minimize(self.d_loss, var_list= self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(args.lr, beta1 = args.beta1).\
            minimize(self.g_loss, var_list= self.g_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.logs_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print("[*] Load SUCCESS")
        else:
            print("[!] Load Failed...")

        for epoch in range(args.epoch):
            data = load('/home/guangyuan/DCGAN/mnist.mat')['X']
            np.random.shuffle(data)

            batch_idxs= min(len(data),args.train_size) // args.batch_size

            for idx in range(batch_idxs):
                st= time.time()
                batch_files = data[idx* args.batch_size: (idx+1)*args.batch_size]
                batch_images = load_data(batch_files).astype(np.float32)
                # save_images(batch_images, [8, 8],
                #             './{}/B_{:2d}_{:4d}'.format("c", epoch, idx))
                batch_labels = load_label(batch_files).astype(np.float32)

                batch_z = np.random.uniform(-1,1, size=(args.batch_size, 100)).astype(np.float32)
                # print("load time:%4.4f" % (time.time() - st))

                _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                                               feed_dict={self.real_data: batch_images,
                                                          self.y: batch_labels, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)
                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                                               feed_dict={self.y: batch_labels, self.z: batch_z}
                                               )
                self.writer.add_summary(summary_str, counter)
                counter += 1

                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f"\
                      %(epoch, idx, batch_idxs, time.time()-start_time)))
                if np.mod(counter,100)==1:
                    self.sample_model(args.sample_dir, epoch, idx)
                if np.mod(counter,1000)==2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name ="dcgan.model"
        model_dir = "%s_%s" %(self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir,model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        model_dir = "%s_%s" %(self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self,sample_dir, epoch, idx):
        data = load('/home/guangyuan/DCGAN/mnist.mat')['X']
        batch_files = data[0:self.batch_size]
        batch_labels = load_label(batch_files).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, size=(self.batch_size, 100)).astype(np.float32)
        img = self.sess.run(self.fake_img,
                               feed_dict = {self.z:batch_z,self.y: batch_labels})

        save_images(img,[int(np.sqrt(self.batch_size)),int(np.sqrt(self.batch_size))],
                    './{}/G_{:2d}_{:4d}'.format(sample_dir,epoch,idx))

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        yp = np.zeros(10*self.batch_size).reshape(self.batch_size, 10)
        for i in range(self.batch_size):
            j = i % 10
            yp[i, j] = 1

        for k in range(100):



            z1 = (np.random.uniform(-1, 1, size=(1, 100)).astype(np.float32)) * np.ones(
                     [int(np.sqrt(self.batch_size)), 100])
            z2 = (np.random.uniform(-1, 1, size=(1, 100)).astype(np.float32)) * np.ones(
                [int(np.sqrt(self.batch_size)), 100])
            batch_z = z1
            for idx, ratio in enumerate(np.linspace(0, 1, 8)):
                z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
                batch_z = np.concatenate([batch_z,z], axis=0)
            batch_z = np.concatenate([batch_z, z2], axis=0)

            img = self.sess.run(self.fake_img,
                                feed_dict={self.z: batch_z, self.y: yp})

            save_images(img, [int(np.sqrt(self.batch_size)),int(np.sqrt(self.batch_size))],
                        './{}/test_G_{:2d}'.format(args.test_dir, k))


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high