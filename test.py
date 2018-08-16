import tensorflow as tf
import os

from read_data import *
from utils import *
from ops import *
from model import *
from model import BATCH_SIZE


def train():

    # global_step is a recorder that records the steps during training.
#    global_step = tf.Variable(0, name='global_step', trainable=False)
    # the path that saves the training logs.

#    CURRENT_DIR=os.getcwd()

#    train_dir = CURRENT_DIR+'/logs'

    # three place holders, 'y' is the constraints, 'images' are the images that feed the discriminator, 'z' is the random variable in the latent space.
#    images_v1 = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='real_images_v1')
#    images_v2 = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='real_images_v2')
#    images_v3 = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='real_images_v3')
#    images_v4 = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1], name='real_images_v4')

#    with tf.variable_scope("for_reuse_scope"):
#        c_v1 = encoder_v1(images_v1)
#        c_v2 = encoder_v2(images_v2)
#        c_v3 = encoder_v3(images_v3)
#        c_v4 = encoder_v4(images_v4)

#        G_v1 = generator_v1(c_v1)
#        G_v2 = generator_v2(c_v2)
#        G_v3 = generator_v3(c_v3)
#        G_v4 = generator_v4(c_v4)

#        D_v1, D_v1_logits = discriminator_v1(images_v1)
#        D_v2, D_v2_logits = discriminator_v2(images_v2)
#        D_v3, D_v3_logits = discriminator_v3(images_v3)
#        D_v4, D_v4_logits = discriminator_v4(images_v4)

#        D_v1_, D_v1_logits_ = discriminator_v1(G_v1, reuse = True)
#        D_v2_, D_v2_logits_ = discriminator_v2(G_v2, reuse = True)
#        D_v3_, D_v3_logits_ = discriminator_v3(G_v3, reuse = True)
#        D_v4_, D_v4_logits_ = discriminator_v4(G_v4, reuse = True)

#        samples_v1 = sampler_v1(c_v1)
#        samples_v2 = sampler_v2(c_v1)
#        samples_v3 = sampler_v3(c_v1)
#        samples_v4 = sampler_v4(c_v1)


    # Define loss functions.
    # The discriminator tries to make the output of real images approximinate one.

#    e_v1_loss = tf.reduce_mean(tf.square(c_v1-c_v4))
#    e_v2_loss = tf.reduce_mean(tf.square(c_v2-c_v4))
#    e_v3_loss = tf.reduce_mean(tf.square(c_v3-c_v4))

#    d_v1_loss_real = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v1_logits, labels=tf.ones_like(D_v1)))
#    d_v2_loss_real = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v2_logits, labels=tf.ones_like(D_v2)))
#    d_v3_loss_real = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v3_logits, labels=tf.ones_like(D_v3)))
#    d_v4_loss_real = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v4_logits, labels=tf.ones_like(D_v4)))

#    d_v1_loss_fake = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v1_logits_, labels=tf.zeros_like(D_v1_)))
#    d_v2_loss_fake = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v2_logits_, labels=tf.zeros_like(D_v2_)))
#    d_v3_loss_fake = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v3_logits_, labels=tf.zeros_like(D_v3_)))
#    d_v4_loss_fake = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v4_logits_, labels=tf.zeros_like(D_v4_)))

#    d_v1_loss = (d_v1_loss_real + d_v1_loss_fake) / 2
#    d_v2_loss = (d_v2_loss_real + d_v2_loss_fake) / 2
#    d_v3_loss = (d_v3_loss_real + d_v3_loss_fake) / 2
#    d_v4_loss = (d_v4_loss_real + d_v4_loss_fake) / 2

#    g_v1_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v1_logits_, labels=tf.ones_like(D_v1_)))
#    g_v2_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v2_logits_, labels=tf.ones_like(D_v2_)))
#    g_v3_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v3_logits_, labels=tf.ones_like(D_v3_)))
#    g_v4_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_v4_logits_, labels=tf.ones_like(D_v4_)))

    # The variables to be updated in generator and discriminator.
#    t_vars = tf.trainable_variables()

#    e_v1_vars = [var for var in t_vars if 'e_v1' in var.name]
#    e_v2_vars = [var for var in t_vars if 'e_v2' in var.name]
#    e_v3_vars = [var for var in t_vars if 'e_v3' in var.name]

#    d_v1_vars = [var for var in t_vars if 'd_v1' in var.name]
#    d_v2_vars = [var for var in t_vars if 'd_v2' in var.name]
#    d_v3_vars = [var for var in t_vars if 'd_v3' in var.name]
#    d_v4_vars = [var for var in t_vars if 'd_v4' in var.name]

#    g_v1_vars = [var for var in t_vars if 'g_v1' in var.name]
#    g_v2_vars = [var for var in t_vars if 'g_v2' in var.name]
#    g_v3_vars = [var for var in t_vars if 'g_v3' in var.name]
#    g_v4_vars = [var for var in t_vars if 'g_v4' in var.name]



#    saver = tf.train.Saver()


    # Adam optimizer is adopted.

#    e_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=e_v1_loss, var_list=e_v1_vars,
#                                                                                  global_step=global_step)
#    e_v2_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=e_v2_loss, var_list=e_v2_vars,
#                                                                                  global_step=global_step)
#    e_v3_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=e_v3_loss, var_list=e_v3_vars,
#                                                                                  global_step=global_step)

#    d_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=d_v1_loss, var_list=d_v1_vars,
#                                                                                  global_step=global_step)
#    d_v2_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=d_v2_loss, var_list=d_v2_vars,
#                                                                                  global_step=global_step)
#    d_v3_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=d_v3_loss, var_list=d_v3_vars,
#                                                                                  global_step=global_step)
#    d_v4_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=d_v4_loss, var_list=d_v4_vars,
#                                                                                  global_step=global_step)

#    g_v1_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v1_loss, var_list=g_v1_vars,
#                                                                                  global_step=global_step)
#    g_v2_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v2_loss, var_list=g_v2_vars,
#                                                                                  global_step=global_step)
#    g_v3_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v3_loss, var_list=g_v3_vars,
#                                                                                  global_step=global_step)
#    g_v4_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(loss=g_v4_loss, var_list=g_v4_vars,
#                                                                                  global_step=global_step)

#    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
#    config = tf.ConfigProto()
#    config.gpu_options.per_process_gpu_memory_fraction = 0.2
#    sess = tf.InteractiveSession(config=config)

#    init = tf.global_variables_initializer()
#    writer = tf.summary.FileWriter(train_dir, sess.graph)

    data_x, X_v1, X_v2, X_v3, X_v4 = read_data()


    sample_X_v1 = X_v1[0: BATCH_SIZE]
    sample_X_v2 = X_v2[0: BATCH_SIZE]
    sample_X_v3 = X_v3[0: BATCH_SIZE]
    sample_X_v4 = X_v4[0: BATCH_SIZE]

#    sess.run(init)

#    for epoch in range(25):
#        batch_idxs = 1093
#        for idx in range(1, batch_idxs):
#            batch_X_v1 = X_v1[idx * BATCH_SIZE: (idx + 1) * 64]
#            batch_X_v2 = X_v2[idx * BATCH_SIZE: (idx + 1) * 64]
#            batch_X_v3 = X_v3[idx * BATCH_SIZE: (idx + 1) * 64]
#            batch_X_v4 = X_v4[idx * BATCH_SIZE: (idx + 1) * 64]

#            sess.run([e_v1_optim], feed_dict = {images_v1: batch_X_v1, images_v4: batch_X_v4})
#            sess.run([e_v2_optim], feed_dict = {images_v2: batch_X_v2, images_v4: batch_X_v4})
#            sess.run([e_v3_optim], feed_dict = {images_v3: batch_X_v3, images_v4: batch_X_v4})



#            print(sample_labels)

            # Update the parameters of D
            #_, summary_str = sess.run([d_optim, d_sum], \
            #                          feed_dict={images: batch_images, \
            #                                     z: batch_z,})
            #writer.add_summary(summary_str, idx + 1)

            # Update the parameters of G
            #_, summary_str = sess.run([g_optim, g_sum], feed_dict={z: batch_z})
            #writer.add_summary(summary_str, idx + 1)

            # Update the parameters of G for the second time to ensure the stability of the network
            #_, summary_str = sess.run([g_optim, g_sum], feed_dict={z: batch_z})
            #writer.add_summary(summary_str, idx + 1)

#            err_e_v1 = e_v1_loss.eval({images_v1: batch_X_v1, images_v4: batch_X_v4})
#            err_e_v2 = e_v2_loss.eval({images_v2: batch_X_v2, images_v4: batch_X_v4})
#            err_e_v3 = e_v3_loss.eval({images_v3: batch_X_v3, images_v4: batch_X_v4})

            #errD_fake = d_loss_fake.eval({z: batch_z})
            #errD_real = d_loss_real.eval({images: batch_images})
            #errG = g_loss.eval({z: batch_z})

#            if idx % 20 == 0:
#                print("Epoch: [%2d] [%4d/%4d] e_v1_loss: %.8f, e_v2_loss: %.8f, e_v3_loss: %.8f" \
#                    % (epoch, idx, batch_idxs, err_e_v1, err_e_v2, err_e_v3))

            #if idx % 100 == 1:
            #    sample = sess.run(samples, feed_dict={z: sample_z})
            #    samples_path = CURRENT_DIR + '/samples/'
            #    save_images(sample, [8, 8], samples_path + 'test_%d_epoch_%d.png' % (epoch, idx))
            #    print 'save down'

            #if idx % 500 == 2:
            #    checkpoint_path = os.path.join(train_dir, 'DCGAN_model.ckpt')
            #    saver.save(sess, checkpoint_path, global_step=idx + 1)


#    for epoch in range(25):
#        batch_idxs = 1093
#        for idx in range(1, batch_idxs):
#            batch_X_v1 = X_v1[idx * BATCH_SIZE: (idx + 1) * 64]
#            batch_X_v2 = X_v2[idx * BATCH_SIZE: (idx + 1) * 64]
#            batch_X_v3 = X_v3[idx * BATCH_SIZE: (idx + 1) * 64]
#            batch_X_v4 = X_v4[idx * BATCH_SIZE: (idx + 1) * 64]

#            sess.run(d_v1_optim, feed_dict = {images_v1: batch_X_v1})
#            sess.run(d_v2_optim, feed_dict = {images_v2: batch_X_v2})
#            sess.run(d_v3_optim, feed_dict = {images_v3: batch_X_v3})
#            sess.run(d_v4_optim, feed_dict = {images_v4: batch_X_v4})

#            sess.run(g_v1_optim, feed_dict = {images_v1: batch_X_v1})
#            sess.run(g_v2_optim, feed_dict = {images_v2: batch_X_v2})
#            sess.run(g_v3_optim, feed_dict = {images_v3: batch_X_v3})
#            sess.run(g_v4_optim, feed_dict = {images_v4: batch_X_v4})

#            sess.run(g_v1_optim, feed_dict = {images_v1: batch_X_v1})
#            sess.run(g_v2_optim, feed_dict = {images_v2: batch_X_v2})
#            sess.run(g_v3_optim, feed_dict = {images_v3: batch_X_v3})
#            sess.run(g_v4_optim, feed_dict = {images_v4: batch_X_v4})

#            err_d_v1_fake = d_v1_loss.eval({images_v1: batch_X_v1})
#            err_d_v2_fake = d_v2_loss.eval({images_v2: batch_X_v2})
#            err_d_v3_fake = d_v3_loss.eval({images_v3: batch_X_v3})
#            err_d_v4_fake = d_v4_loss.eval({images_v4: batch_X_v4})

#            err_g_v1_fake = g_v1_loss.eval({images_v1: batch_X_v1})
#            err_g_v2_fake = g_v2_loss.eval({images_v2: batch_X_v2})
#            err_g_v3_fake = g_v3_loss.eval({images_v3: batch_X_v3})
#            err_g_v4_fake = g_v4_loss.eval({images_v4: batch_X_v4})


#            if idx % 20 == 0:
                #print("Epoch: [%2d] [%4d/%4d] d_v1_loss: %.8f, d_v2_loss: %.8f, d_v3_loss: %8f, d_v4_loss: %.8f, g_v1_loss: %.8f, g_v2_loss: %.8f, g_v3_loss: %.8f, g_v4_loss: %.8f" % (epoch, idx, batch_idxs, err_d_v1_fake, err_d_v2_fake, err_d_v3_fake, err_d_v4_fake, err_g_v1_fake, err_g_v2_fake, err_g_v3_fake, err_g_v4_fake))
#                print(
#                "Epoch: [%2d] [%4d/%4d] d_v1_loss: %.8f, d_v2_loss: %.8f, d_v3_loss: %8f, d_v4_loss: %.8f, g_v1_loss: %.8f, g_v2_loss: %.8f, g_v3_loss: %.8f, g_v4_loss: %.8f" % (
#                epoch, idx, batch_idxs, err_d_v1_fake, err_d_v2_fake, err_d_v3_fake, err_d_v4_fake, err_g_v1_fake,
#                err_g_v2_fake, err_g_v3_fake, err_g_v4_fake))


#            if idx % 100 ==1:
#                sample_v1 = sess.run(samples_v1, feed_dict = {images_v1: sample_images})
#                sample_v2 = sess.run(samples_v2, feed_dict = {images_v1: sample_images})
#                sample_v3 = sess.run(samples_v3, feed_dict = {images_v1: sample_images})
#                sample_v4 = sess.run(samples_v4, feed_dict = {images_v1: sample_images})
    epoch = 1
    idx = 1

    samples_path = os.getcwd()+'/test_images/'
    save_images(sample_X_v1, [8, 8], samples_path + 'test_v1_%d_epoch_%d.png' % (epoch, idx))
    save_images(sample_X_v2, [8, 8], samples_path + 'test_v2_%d_epoch_%d.png' % (epoch, idx))
    save_images(sample_X_v3, [8, 8], samples_path + 'test_v3_%d_epoch_%d.png' % (epoch, idx))
    save_images(sample_X_v4, [8, 8], samples_path + 'test_v4_%d_epoch_%d.png' % (epoch, idx))




#    sess.close()

if __name__ == '__main__':
    train()
