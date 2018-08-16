import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import os
from read_data import *
from utils import *
from ops import *
from model import *
from model import BATCH_SIZE
from full_metric import *


def eval():
    # global_step is a recorder that records the steps during training.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # the path that saves the training logs.
    CURRENT_DIR = os.getcwd()

    train_dir = CURRENT_DIR + '/logs'

    # Place holders, images_v1, images_v2, images_v3, images_v4 are taken from 4 views, z_v1, z_v2, z_v3, z_v4 are
    images_v1 = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images_v1')
    images_v2 = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images_v2')
    images_v3 = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3], name='real_images_v3')

    with tf.variable_scope("for_reuse_scope"):
        c_v1 = encoder_v1(images_v1)
        c_v2 = encoder_v2(images_v2)
        # c_v3 = encoder_v3(images_v3)
        # G_v2 = generator_v2(c_v2)

        # G_v1 = generator_v1(z_v1, c_v1)
        G_v1 = generator_v1(c_v1)
        #D_v1, D_v1_logits, Q_v1, Q_v1_logits = discriminator_v1(images_v1)
        #D_v1_, D_v1_logits_, Q_v1_, Q_v1_logits_ = discriminator_v1(G_v1, reuse=True)

        # samples_v1_to_v1 = sampler_v1(z_v1, c_v1)
        # samples_v2_to_v1 = sampler_v1(z_v1, c_v2)
        samples_v1_to_v1 = sampler_v1(c_v1)
        samples_v2_to_v1 = sampler_v1(c_v2)

    # Reading checkpoints, sess, saver are needed.
    print("Reading checkpoints...")
    ckpt = tf.train.get_checkpoint_state(train_dir)
    saver = tf.train.Saver()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(train_dir, sess.graph)

    #X_v1, X_v2 = read_data()
    X_v2 = read_test_data()

    sess.run(init)
    # Restore variables from models.
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(train_dir, ckpt_name))
        print "reload ok!"
    else:
        print "reload failed..."
    # ssim_sum_v2_to_v1 = 0.
    # psnr_sum_v2_to_v1 = 0.
    # vifp_sum_v2_to_v1 = 0.
    # ssim_list = []
    # psnr_list = []
    # for idx in range(4):
    #     sample_images_v1 = (X_v1[idx * BATCH_SIZE: (idx + 1) * 64] - 127.5) / 127.5
    #     sample_images_v2 = (X_v2[idx * BATCH_SIZE: (idx + 1) * 64] - 127.5) / 127.5
    #     sample_v2 = sess.run(samples_v2_to_v1, feed_dict={images_v2: sample_images_v2})
    #
    #     sample_images_v1_255 = sample_images_v1*127.5+127.5
    #     sample_images_v2_255 = sample_images_v2*127.5+127.5
    #     sample_v2_255 = sample_v2*127.5+127.5
    #
    #     ssim_v2_to_v1 = ssim(sample_images_v1, sample_v2)
    #     ssim_list.append(ssim_sum_v2_to_v1)
    #     ssim_sum_v2_to_v1 += ssim_v2_to_v1
    #
    #     vifp_v2_to_v1 = vifp(sample_images_v1, sample_v2)
    #     vifp_sum_v2_to_v1 += vifp_v2_to_v1
    #
    #     psnr_v2_to_v1 = psnr(sample_images_v1_255, sample_v2_255)
    #     psnr_list.append(psnr_sum_v2_to_v1)
    #     psnr_sum_v2_to_v1 += psnr_v2_to_v1
    #
    #     samples_path = os.getcwd() + '/eval/'
    #     save_images(sample_images_v1, [8, 8], samples_path + 'v1_original_%d.png' % idx)
    #     save_images(sample_images_v2, [8, 8], samples_path + 'v2_original_%d.png' % idx)
    #     save_images(sample_v2, [8, 8], samples_path + 'v2_to_v1_%d.png'%idx)
    #
    # avg_ssim_sum_v2_to_v1 = ssim_sum_v2_to_v1/4.0
    # avg_psnr_sum_v2_to_v1 = psnr_sum_v2_to_v1/4.0
    # avg_vifp_sum_v2_to_v1 = vifp_sum_v2_to_v1/4.0
    #
    # var_ssim = np.var(ssim_list)
    # var_psnr = np.var(psnr_list)
    #
    # std_ssim = np.sqrt(var_ssim)
    # std_psnr = np.sqrt(var_psnr)
    #
    # print "Average ssim v2 to v1: %.8f, std: %.8f" % (avg_ssim_sum_v2_to_v1, std_ssim)
    # print "Average psnr v2 to v1: %.8f, std: %.8f" % (avg_psnr_sum_v2_to_v1, std_psnr)
    # print "Average vifp v2 to v1: %.8f" % (avg_vifp_sum_v2_to_v1)
    sample_images_v2 = (X_v2 - 127.5) / 127.5
    sample_v2 = sess.run(samples_v2_to_v1, feed_dict={images_v2: sample_images_v2})
    samples_path = os.getcwd() + '/eval/'
    save_images(sample_v2, [8, 8], samples_path + 'v2_to_v1_result.png')


    sess.close()


if __name__ == '__main__':
    eval()