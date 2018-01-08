from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
from skimage import feature
# Building conditional discriminator and conditional generator
# Defining kinds of loss criterions

def cgenerator(z,y,option,reuse = False, name="cg"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        yl = tf.reshape(y, [option.batch_size,1,1,10])
        z = tf.concat([z,y],1)

        h0 = tf.nn.relu(batch_norm(linear(z, 1024,name = "g_h0_lin" ),name="g_bn0"))
        h0 = tf.concat([h0,y],1)

        h1 = tf.nn.relu(batch_norm(
            linear(h0,option.gf_dim*2*int(option.image_size/4)*int(option.image_size/4),name = "g_h1_lin"),name="g_bn1"))
        h1= tf.reshape(h1, [option.batch_size,int(option.image_size/4),int(option.image_size/4),option.gf_dim*2])
        h1 = conv_concat(h1,yl)

        h2 = tf.nn.relu(batch_norm(deconv2d(h1,option.gf_dim*2, ks=5,s=2,name="g_h2_deconv"),"g_bn2"))
        h2 = conv_concat(h2,yl)

        h3 = tf.nn.sigmoid(deconv2d(h2,option.output_c_dim,5,2, name="g_h3_deconv"))
        return h3


def cdiscriminator(image, y,option, reuse=False, name="cdis"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        yl = tf.reshape(y, [option.batch_size,1,1,10])

        x = conv_concat(image, yl)

        h0 = lrelu(conv2d(x,1+10,5,2,name="d_h0_conv"))
        h0 = conv_concat(h0,yl)

        h1 =lrelu(batch_norm(conv2d(h0,64+10,5,2,name="d_h1_conv"),"d_bn1"))
        h1 = tf.reshape(h1,[option.batch_size,-1])
        h1 =tf.concat([h1,y], 1)

        h2 = lrelu(batch_norm(linear(h1, 1024, name="d_h2_linear"),'d_bn2'))
        h2= tf.concat([h2,y],1)

        h3 = linear(h2,1,name="d_h3_linear")

        return h3

def abs(x,y):
    return tf.reduce_mean(tf.abs(x,y))

def mae(x,y):
    return tf.reduce_mean((x-y)**2)

def sce(x,y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y))



