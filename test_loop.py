import tensorflow as tf
import model as model
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import model_zoo
import tfwrapper.utils as tf_utils
import image_utils
import cv2

import utils

IMAGE_SIZE = [224, 224]

def augmentation_function(images, labels):

    # TODO: Create kwargs with augmentation options

    if images.ndim == 3:
        print('so this happened')
        images = images[np.newaxis, ...]
        labels = labels[np.newaxis, ...]

    new_images = []
    new_labels = []
    num_images = images.shape[0]

    for ii in range(num_images):

        random_angle = np.random.uniform(-15, 15)
        img = np.squeeze(images[ii,...])
        lbl = np.squeeze(labels[ii,...])

        img = image_utils.rotate_image(img, random_angle)
        lbl = image_utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

        # cv2.imshow('image', image_utils.convert_to_uint8(img))
        # cv2.imshow('labels', image_utils.convert_to_uint8(lbl))
        # cv2.waitKey(0)

        new_images.append(img[..., np.newaxis])
        new_labels.append(lbl[...])

    sampled_image_batch = np.asarray(new_images)
    sampled_label_batch = np.asarray(new_labels)

    if sampled_image_batch.ndim == 3:
        print('so that happened')
        sampled_image_batch = sampled_image_batch[np.newaxis, ...]
        sampled_label_batch = sampled_label_batch[np.newaxis, ...]

    return sampled_image_batch, sampled_label_batch

def placeholder_inputs(batch_size):

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='images')
    labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size,IMAGE_SIZE[0], IMAGE_SIZE[1], 1), name='labels')
    return images_placeholder, labels_placeholder

# data = h5py.File('newdata_288x288.hdf5', 'r')
data = h5py.File('newdata_224x224.hdf5', 'r')
# images_val = data['images_test'][:]
# labels_val = data['masks_test'][:]
images_val = data['images_test'][:]
labels_val = data['masks_test'][:]


images_val = np.reshape(images_val, (images_val.shape[0], IMAGE_SIZE[0], IMAGE_SIZE[1], 1))


images_placeholder, labels_placeholder = placeholder_inputs(1)

# inference_handle = model_zoo.lisa_net_deeper
inference_handle = model_zoo.unet_bn

mask, softmax = model.predict(images_placeholder, inference_handle)

saver = tf.train.Saver()

init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)
    # saver.restore(sess, tf.train.latest_checkpoint('./acdc_logdir/lisa_net_deeper_mom0.9_sched_reg0.00000_lr0.1_aug_newbn'))
    # saver.restore(sess, tf.train.latest_checkpoint('./acdc_logdir/lisa_net_deeper_adam_sched_reg0.00005_lr0.001_aug_refunweighted'))
    # saver.restore(sess, tf.train.latest_checkpoint('./acdc_logdir/unet_bn_adam_reg0.00000_lr0.01_aug_2'))

    checkpoint_path = utils.get_latest_model_checkpoint_path('acdc_logdir/unet_bn_merged_wenjia_new', 'model_best_dice.ckpt')
    saver.restore(sess, checkpoint_path)

    ind = -1

    while True:

        # ind = np.random.randint(images_val.shape[0])
        ind += 1

        x = images_val[np.newaxis, ind,...]
        x_tensor = np.tile(images_val[ind,...], (1,1,1,1))

        y = labels_val[ind,...]

        # y_tensor = np.reshape(y, [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
        y_tensor = np.tile(np.reshape(y, [IMAGE_SIZE[0], IMAGE_SIZE[1], 1]), (1,1))

        # x, y_tensor = augmentation_function(x, y_tensor)

        print(x.min())
        print(x.max())
        print(x.mean())

        # x = image_utils.normalise_image(x)

        print(x.min())
        print(x.max())
        print(x.mean())

        print('---')

        feed_dict = {
            images_placeholder: x_tensor,
        }

        mask_out, logits_out = sess.run([mask, softmax], feed_dict=feed_dict)

        mask_out = mask_out[np.newaxis,0,...]
        logits_out = logits_out[np.newaxis,0,...]

        logits_crf = np.squeeze(logits_out)
        logits_crf = np.transpose(logits_crf, (2, 0, 1))

        d = dcrf.DenseCRF2D(IMAGE_SIZE[0], IMAGE_SIZE[1], 4)  # width, height, nlabels
        U = unary_from_softmax(logits_crf)
        U = U.reshape((4, -1))
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3)
        # d.addPairwiseBilateral()

        Q = d.inference(50)

        # Find out the most probable class for each pixel.
        MAP = np.argmax(Q, axis=0)
        MAP = np.reshape(MAP, [IMAGE_SIZE[0], IMAGE_SIZE[1]])

        fig = plt.figure()
        ax1 = fig.add_subplot(241)
        ax1.imshow(np.squeeze(x), cmap='gray')
        ax2 = fig.add_subplot(242)
        ax2.imshow(np.squeeze(mask_out))
        ax3 = fig.add_subplot(243)
        ax3.imshow(np.squeeze(MAP))
        ax4 = fig.add_subplot(244)
        ax4.imshow(np.squeeze(y))

        logits_c0 = np.squeeze(logits_out[0,...,0])
        logits_c1 = np.squeeze(logits_out[0,...,1])
        logits_c2 = np.squeeze(logits_out[0,...,2])
        logits_c3 = np.squeeze(logits_out[0,...,3])

        ax5 = fig.add_subplot(245)
        ax5.imshow(logits_c0)

        ax6 = fig.add_subplot(246)
        ax6.imshow(logits_c1)

        ax7 = fig.add_subplot(247)
        ax7.imshow(logits_c2)

        ax8 = fig.add_subplot(248)
        ax8.imshow(logits_c3)

        plt.show()



