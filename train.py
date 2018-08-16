import tensorflow as tf
import os

import numpy as np
from read_data import *
from utils import *
from ops import *
from model import *
from model import BATCH_SIZE


def train():

    # global_step is a recorder that records the steps during training.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # the path that saves the training logs.

    CURRENT_DIR=os.getcwd()

    train_dir = CURRENT_DIR+'/logs'

    # Place holders, images_v1, images_v2, images_v3, images_v4 are taken from 4 views, z_v1, z_v2, z_v3, z_v4 are
    # variables in latent space.
    images_v1 = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images_v1')
    #z_v1 = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='z_v1')
    images_v2 = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images_v2')
    #images_v3 = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images_v3')

    with tf.variable_scope("for_reuse_scope"):
        c_v1 = encoder_v1(images_v1)
        c_v2 = encoder_v2(images_v2)
        #c_v3 = encoder_v3(images_v3)
        #G_v2 = generator_v2(c_v2)

        #G_v1 = generator_v1(z_v1, c_v1)
        G_v1 = generator_v1(c_v1)
        D_v1, D_v1_logits, Q_v1, Q_v1_logits = discriminator_v1(images_v1)
        D_v1_, D_v1_logits_, Q_v1_, Q_v1_logits_ = discriminator_v1(G_v1, reuse=True)

        #samples_v1_to_v1 = sampler_v1(z_v1, c_v1)
        #samples_v2_to_v1 = sampler_v1(z_v1, c_v2)
        samples_v1_to_v1 = sampler_v1(c_v1)
        samples_v2_to_v1 = sampler_v1(c_v2)
        #samples_v3_to_v1 = sampler_v1(c_v3)

    # Define loss functions.
    #g_v2_loss_e = tf.reduce_mean(tf.square(images_v2 - G_v2))
    #e_v1_loss = tf.reduce_mean(tf.square(c_v1 - c_v2))
    #e_v2_loss = tf.reduce_mean(tf.square(images_v2 - G_v2))
    g_v1_loss_e = tf.reduce_mean(tf.square(images_v1-G_v1))
    e_v1_loss = tf.reduce_mean(tf.square(images_v1-G_v1))
    e_v2_loss = tf.reduce_mean(tf.square(c_v2 - c_v1))
    #e_v3_loss = tf.reduce_mean(tf.square(c_v3 - c_v1))

    d_v1_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v1_logits, labels=tf.ones_like(D_v1)))
    d_v1_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v1_logits_, labels=tf.zeros_like(D_v1_)))
    d_v1_loss = (d_v1_loss_real + d_v1_loss_fake) / 2
    g_v1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v1_logits_, labels=tf.ones_like(D_v1_)))
    q_v1_loss = tf.reduce_mean(tf.square(Q_v1_logits_ - c_v1))
    #q_v1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Q_v1_logits_, labels=c_v1))


    # The variables to be updated in generator and discriminator.
    t_vars = tf.trainable_variables()

    e_v1_vars = [var for var in t_vars if 'e_v1' in var.name]
    e_v2_vars = [var for var in t_vars if 'e_v2' in var.name]
    #e_v3_vars = [var for var in t_vars if 'e_v3' in var.name]

    d_v1_vars = [var for var in t_vars if 'd_v1' in var.name]
    q_v1_vars = [var for var in t_vars if 'q_v1' in var.name]

    g_v1_vars = [var for var in t_vars if 'g_v1' in var.name]
    #g_v2_vars = [var for var in t_vars if 'g_v2' in var.name]


    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(train_dir)
    saver = tf.train.Saver()


    # Adam optimizer is adopted.

    e_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=e_v1_loss, var_list=e_v1_vars,
                                                                                 global_step=global_step)
    e_v2_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=e_v2_loss, var_list=e_v2_vars,
                                                                                  global_step=global_step)
    #e_v3_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=e_v3_loss, var_list=e_v3_vars,
    #                                                                              global_step=global_step)
    #g_v2_optm_e = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v2_loss_e, var_list=g_v2_vars,
    #                                                                               global_step=global_step)
    g_v1_optm_e = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v1_loss_e, var_list=g_v1_vars,
                                                                                   global_step=global_step)

    d_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=d_v1_loss, var_list=d_v1_vars,
                                                                                 global_step=global_step)
    g_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v1_loss, var_list=g_v1_vars,
                                                                                 global_step=global_step)
    q_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=0.0005*q_v1_loss, var_list=g_v1_vars +
                                                                        d_v1_vars+q_v1_vars, global_step=global_step)


    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(train_dir, sess.graph)

    #X_v1, X_v2, X_v3 = read_data()
    X_v1, X_v2 = read_data()

    sample_images_v1 = (X_v1[0: BATCH_SIZE] -127.5)/127.5
    sample_images_v2 = (X_v2[0: BATCH_SIZE] -127.5)/127.5
    #sample_images_v3 = (X_v3[0: BATCH_SIZE] - 127.5) / 127.5
    save_images(sample_images_v1, [8, 8], os.getcwd()+'/samples/real_image1.png')
    save_images(sample_images_v2, [8, 8], os.getcwd() + '/samples/real_image2.png')
    #save_images(sample_images_v3, [8, 8], os.getcwd() + '/samples/real_image3.png')
    #sample_z_v1 = np.random.uniform(-1, 1, (BATCH_SIZE, 10))

    sess.run(init)
    # Restore variables from models.
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(train_dir, ckpt_name))
        print "reload ok!"
    else:
        print "reload failed..."

    for epoch in range(10):
        batch_idxs = len(X_v1) // BATCH_SIZE
        for idx in range(1, batch_idxs):
            batch_X_v1 = (X_v1[idx * BATCH_SIZE: (idx + 1) * 64] -127.5)/127.5
            batch_X_v2 = (X_v2[idx * BATCH_SIZE: (idx + 1) * 64] -127.5)/127.5
            #batch_X_v3 = (X_v3[idx * BATCH_SIZE: (idx + 1) * 64] - 127.5) / 127.5

            sess.run([e_v1_optim], feed_dict={images_v1: batch_X_v1})
            sess.run([e_v2_optim], feed_dict={images_v1: batch_X_v1, images_v2: batch_X_v2})
            #sess.run([e_v3_optim], feed_dict={images_v1: batch_X_v1, images_v3: batch_X_v3})
            sess.run([g_v1_optm_e], feed_dict={images_v1: batch_X_v1})

            err_e_v1 = e_v1_loss.eval({images_v1: batch_X_v1, images_v2: batch_X_v2})
            err_e_v2 = e_v2_loss.eval({images_v1: batch_X_v1, images_v2: batch_X_v2})
            #err_e_v3 = e_v3_loss.eval({images_v1: batch_X_v1, images_v3: batch_X_v3})
            err_de_v1 = g_v1_loss_e.eval({images_v1: batch_X_v1, images_v2: batch_X_v2})

            if idx % 20 == 0:
                print("Epoch: [%2d] [%4d/%4d] e_v1_loss: %.8f, e_v2_loss: %.8f, g_v1_loss: %.8f" \
                      % (epoch, idx, batch_idxs, err_e_v1, err_e_v2, err_de_v1))

            if idx%1000 == 1:
                sample = sess.run(G_v1, feed_dict={images_v1: sample_images_v1})
                save_images(sample, [8, 8], os.getcwd()+'/test/test_%d_epoch_%d.png'%(epoch, idx))


    for epoch in range(10):
        batch_idxs = len(X_v1)//BATCH_SIZE
        for idx in range(1, batch_idxs):
            batch_X_v1 = (X_v1[idx * BATCH_SIZE: (idx + 1) * 64] -127.5)/127.5
            #batch_z_v1 = np.random.uniform(-1, 1, (BATCH_SIZE, 10))

            sess.run(d_v1_optim, feed_dict={images_v1: batch_X_v1})
            sess.run(g_v1_optim, feed_dict={images_v1: batch_X_v1})
            sess.run(g_v1_optim, feed_dict={images_v1: batch_X_v1})
            sess.run(q_v1_optim, feed_dict={images_v1: batch_X_v1})

            err_d_v1_fake = d_v1_loss.eval({images_v1: batch_X_v1})
            err_g_v1_fake = g_v1_loss.eval({images_v1: batch_X_v1})
            err_q_v1_fake = q_v1_loss.eval({images_v1: batch_X_v1})

            if idx % 20 == 0:
                print(
                "Epoch: [%2d] [%4d/%4d]  d_v1_loss: %.8f, g_v1_loss: %.8f, q_v1_loss: %.8f" % (
                epoch, idx, batch_idxs, err_d_v1_fake, err_g_v1_fake, err_q_v1_fake))

            if idx % 1000 ==1:
                sample_v1 = sess.run(samples_v1_to_v1, feed_dict={images_v1: sample_images_v1})
                sample_v2 = sess.run(samples_v2_to_v1, feed_dict={images_v2: sample_images_v2})
                #sample_v3 = sess.run(samples_v3_to_v1, feed_dict={images_v3: sample_images_v3})

                samples_path = os.getcwd()+'/samples/'
                save_images(sample_v1, [8, 8], samples_path + 'test_%d_epoch_%d_v1.png' % (epoch, idx))
                save_images(sample_v2, [8, 8], samples_path + 'test_%d_epoch_%d_v2.png' % (epoch, idx))
                #save_images(sample_v3, [8, 8], samples_path + 'test_%d_epoch_%d_v3.png' % (epoch, idx))

            if idx % 2000 == 2:
                checkpoint_path = os.path.join(train_dir, 'GAN_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=idx + 1)

    sess.close()

if __name__ == '__main__':
    train()
