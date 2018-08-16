import tensorflow as tf
from ops import *

BATCH_SIZE = 64
OUTPUT_SIZE = 64
C_DIM = 100


def generator_v1(z, is_train=True, name='generator_v1'):

    with tf.name_scope(name):

        s2, s4, s8, s16 = OUTPUT_SIZE/2, OUTPUT_SIZE/4, OUTPUT_SIZE /8, OUTPUT_SIZE/ 16
        #z = tf.concat([c, z], 1, name='g_v1_c_concat_z')

        h1 = tf.reshape(fully_connected(z, 64*8*s16*s16, 'g_v1_fc1'), [-1, s16, s16, 64*8], name='g_v1_reshape')
        h1 = relu(batch_norm_layer(h1, name='g_v1_bn1', is_train=is_train))

        h2 = deconv2d(h1, [BATCH_SIZE, s8, s8, 64*4], name='g_v1_deconv2d1')
        h2 = relu(batch_norm_layer(h2, name='g_v1_bn2', is_train=is_train))

        h3 = deconv2d(h2, [BATCH_SIZE, s4, s4, 64*2], name='g_v1_deconv2d2')
        h3 = relu(batch_norm_layer(h3, name='g_v1_bn3', is_train=is_train))

        h4 = deconv2d(h3, [BATCH_SIZE, s2, s2, 64*1], name='g_v1_deconv2d3')
        h4 = relu(batch_norm_layer(h4, name='g_v1_bn4', is_train=is_train))

        h5 = deconv2d(h4, [BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3], name='g_v1_deconv2d4')

        return tf.nn.tanh(h5)


def generator_v2(z, is_train=True, name='generator_v2'):

    with tf.name_scope(name):

        s2, s4, s8, s16 = OUTPUT_SIZE/2, OUTPUT_SIZE/4, OUTPUT_SIZE /8, OUTPUT_SIZE/ 16

        h1 = tf.reshape(fully_connected(z, 64*8*s16*s16, 'g_v2_fc1'), [-1, s16, s16, 64*8], name='g_v2_reshape')
        h1 = relu(batch_norm_layer(h1, name='g_v2_bn1', is_train=is_train))

        h2 = deconv2d(h1, [BATCH_SIZE, s8, s8, 64*4], name='g_v2_deconv2d1')
        h2 = relu(batch_norm_layer(h2, name='g_v2_bn2', is_train=is_train))

        h3 = deconv2d(h2, [BATCH_SIZE, s4, s4, 64*2], name='g_v2_deconv2d2')
        h3 = relu(batch_norm_layer(h3, name='g_v2_bn3', is_train=is_train))

        h4 = deconv2d(h3, [BATCH_SIZE, s2, s2, 64*1], name='g_v2_deconv2d3')
        h4 = relu(batch_norm_layer(h4, name='g_v2_bn4', is_train=is_train))

        h5 = deconv2d(h4, [BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE, 3], name='g_v2_deconv2d4')

        return tf.nn.tanh(h5)


def discriminator_v1(image, reuse=False, name='discriminator_v1'):
    with tf.name_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = lrelu(conv2d(image, 64, name='d_v1_h0_conv'), name='d_v1_h0_lrelu')

        h1 = lrelu(batch_norm_layer(conv2d(h0, 64 * 2, name='d_v1_h1_conv'), name='d_v1_h1_bn'), name='d_v1_h1_lrelu')

        h2 = lrelu(batch_norm_layer(conv2d(h1, 64 * 4, name='d_v1_h2_conv'), name='d_v1_h2_bn'), name='d_v1_h2_lrelu')

        h3 = lrelu(batch_norm_layer(conv2d(h2, 64 * 8, name='d_v1_h3_conv'), name='d_v1_h3_bn'), name='d_v1_h3_lrelu')
        h3 = tf.reshape(h3, [BATCH_SIZE, -1])

        d = fully_connected(h3, 1, 'd_v1_h4_fc')
        q = lrelu(batch_norm_layer(fully_connected(h3, 1024,'q_v1_fc1'), name='q_v1_bn1'),name='q_v1_lrelu1')
        q = fully_connected(q, C_DIM, 'q_v1_fc2')

        return tf.nn.sigmoid(d), d, tf.nn.sigmoid(q), q


def encoder_v1(image, train=True, name='encoder_v1'):
    with tf.name_scope(name):

        h0 = lrelu(conv2d(image, 64, name='e_v1_h0_conv'), name='e_v1_h0_lrelu')
        h1 = lrelu(batch_norm_layer(conv2d(h0, 64 * 2, name='e_v1_h1_conv'), name='e_v1_h1_bn'), name='e_v1_h1_lrelu')
        h2 = lrelu(batch_norm_layer(conv2d(h1, 64 * 4, name='e_v1_h2_conv'), name='e_v1_h2_bn'), name='e_v1_h2_lrelu')
        h3 = lrelu(batch_norm_layer(conv2d(h2, 64 * 8, name='e_v1_h3_conv'), name='e_v1_h3_bn'), name='e_v1_h3_lrelu')
        h4 = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]), C_DIM, 'e_v1_h4_fc')

        return tf.nn.sigmoid(h4)
        #return h4


def encoder_v2(image, train=True, name='encoder_v2'):
    with tf.name_scope(name):

        h0 = lrelu(conv2d(image, 64, name='e_v2_h0_conv'), name='e_v2_h0_lrelu')
        h1 = lrelu(batch_norm_layer(conv2d(h0, 64 * 2, name='e_v2_h1_conv'), name='e_v2_h1_bn'), name='e_v2_h1_lrelu')
        h2 = lrelu(batch_norm_layer(conv2d(h1, 64 * 4, name='e_v2_h2_conv'), name='e_v2_h2_bn'), name='e_v2_h2_lrelu')
        h3 = lrelu(batch_norm_layer(conv2d(h2, 64 * 8, name='e_v2_h3_conv'), name='e_v2_h3_bn'), name='e_v2_h3_lrelu')
        h4 = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]), C_DIM, 'e_v2_h4_fc')

        return tf.nn.sigmoid(h4)
        #return h4


def encoder_v3(image, train=True, name='encoder_v3'):
    with tf.name_scope(name):

        h0 = lrelu(conv2d(image, 64, name='e_v3_h0_conv'), name='e_v3_h0_lrelu')
        h1 = lrelu(batch_norm_layer(conv2d(h0, 64 * 2, name='e_v3_h1_conv'), name='e_v3_h1_bn'), name='e_v3_h1_lrelu')
        h2 = lrelu(batch_norm_layer(conv2d(h1, 64 * 4, name='e_v3_h2_conv'), name='e_v3_h2_bn'), name='e_v3_h2_lrelu')
        h3 = lrelu(batch_norm_layer(conv2d(h2, 64 * 8, name='e_v3_h3_conv'), name='e_v3_h3_bn'), name='e_v3_h3_lrelu')
        h4 = fully_connected(tf.reshape(h3, [BATCH_SIZE, -1]), C_DIM, 'e_v3_h4_fc')

        return tf.nn.sigmoid(h4)
        #return h4


def sampler_v1(z, is_train=False, name='sampler_v1'):
    with tf.name_scope(name):
        tf.get_variable_scope().reuse_variables()
        return generator_v1(z, is_train=is_train)

