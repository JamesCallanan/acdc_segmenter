from tfwrapper import layers
import tensorflow as tf

NUM_CLASSES = 4

def unet_bn(images, training):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)

    pool1 = layers.max_pool_layer2d(conv1_2)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)

    pool2 = layers.max_pool_layer2d(conv2_2)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)

    pool3 = layers.max_pool_layer2d(conv3_2)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)

    pool4 = layers.max_pool_layer2d(conv4_2)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training)

    upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=3, name='concat4')

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)

    upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)

    upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)

    upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=NUM_CLASSES, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1), activation=layers.no_activation, training=training)

    return pred


def unet_bn_translated_to_3D(images, training):

    conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=64, kernel_size=(3,3,1), training=training)
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64, kernel_size=(3,3,1), training=training)

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=128, kernel_size=(3,3,1), training=training)
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128, kernel_size=(3,3,1), training=training)

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=256, kernel_size=(3,3,1), training=training)
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256, kernel_size=(3,3,1), training=training)

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=512, kernel_size=(3,3,1), training=training)
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512, kernel_size=(3,3,1), training=training)

    pool4 = layers.max_pool_layer3d(conv4_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', num_filters=1024, kernel_size=(3,3,1), training=training)
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, kernel_size=(3,3,1), training=training)

    upconv4 = layers.deconv3D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=512, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=4, name='concat4')

    conv6_1 = layers.conv3D_layer_bn(concat4, 'conv6_1', num_filters=512, kernel_size=(3,3,1), training=training)
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=512, kernel_size=(3,3,1), training=training)

    upconv3 = layers.deconv3D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=256, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=4, name='concat3')

    conv7_1 = layers.conv3D_layer_bn(concat3, 'conv7_1', num_filters=256, kernel_size=(3,3,1), training=training)
    conv7_2 = layers.conv3D_layer_bn(conv7_1, 'conv7_2', num_filters=256, kernel_size=(3,3,1), training=training)

    upconv2 = layers.deconv3D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=128, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=4, name='concat2')

    conv8_1 = layers.conv3D_layer_bn(concat2, 'conv8_1', num_filters=128, kernel_size=(3,3,1), training=training)
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=128, kernel_size=(3,3,1), training=training)

    upconv1 = layers.deconv3D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=64, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=4, name='concat1')

    conv9_1 = layers.conv3D_layer_bn(concat1, 'conv9_1', num_filters=64, kernel_size=(3,3,1), training=training)
    conv9_2 = layers.conv3D_layer_bn(conv9_1, 'conv9_2', num_filters=64, kernel_size=(3,3,1), training=training)

    pred = layers.conv3D_layer_bn(conv9_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1,1), activation=layers.no_activation, training=training)

    return pred


def unet_bn_translated_to_3D_half(images, training):

    conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=64//2, kernel_size=(3,3,1), training=training)
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64//2, kernel_size=(3,3,1), training=training)

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=128//2, kernel_size=(3,3,1), training=training)
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128//2, kernel_size=(3,3,1), training=training)

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=256//2, kernel_size=(3,3,1), training=training)
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256//2, kernel_size=(3,3,1), training=training)

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=512//2, kernel_size=(3,3,1), training=training)
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512//2, kernel_size=(3,3,1), training=training)

    pool4 = layers.max_pool_layer3d(conv4_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', num_filters=1024//2, kernel_size=(3,3,1), training=training)
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=1024//2, kernel_size=(3,3,1), training=training)

    upconv4 = layers.deconv3D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=512//2, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=4, name='concat4')

    conv6_1 = layers.conv3D_layer_bn(concat4, 'conv6_1', num_filters=512//2, kernel_size=(3,3,1), training=training)
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=512//2, kernel_size=(3,3,1), training=training)

    upconv3 = layers.deconv3D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=256//2, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=4, name='concat3')

    conv7_1 = layers.conv3D_layer_bn(concat3, 'conv7_1', num_filters=256//2, kernel_size=(3,3,1), training=training)
    conv7_2 = layers.conv3D_layer_bn(conv7_1, 'conv7_2', num_filters=256//2, kernel_size=(3,3,1), training=training)

    upconv2 = layers.deconv3D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=128//2, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=4, name='concat2')

    conv8_1 = layers.conv3D_layer_bn(concat2, 'conv8_1', num_filters=128//2, kernel_size=(3,3,1), training=training)
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=128//2, kernel_size=(3,3,1), training=training)

    upconv1 = layers.deconv3D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=64//2, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=4, name='concat1')

    conv9_1 = layers.conv3D_layer_bn(concat1, 'conv9_1', num_filters=64//2, kernel_size=(3,3,1), training=training)
    conv9_2 = layers.conv3D_layer_bn(conv9_1, 'conv9_2', num_filters=64//2, kernel_size=(3,3,1), training=training)

    pred = layers.conv3D_layer_bn(conv9_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1,1), activation=layers.no_activation, training=training)

    return pred



def unet_bn_2D3D(images, training):

    conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=64, kernel_size=(3,3,1), training=training)
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64, kernel_size=(3,3,1), training=training)

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=128, kernel_size=(3,3,1), training=training)
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128, kernel_size=(3,3,1), training=training)

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=256, kernel_size=(3,3,3), training=training)
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256, kernel_size=(3,3,3), training=training)

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=512, kernel_size=(3,3,3), training=training)
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512, kernel_size=(3,3,3), training=training)

    pool4 = layers.max_pool_layer3d(conv4_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', num_filters=1024, kernel_size=(3,3,3), training=training)
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, kernel_size=(3,3,3), training=training)

    upconv4 = layers.deconv3D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=512, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=4, name='concat4')

    conv6_1 = layers.conv3D_layer_bn(concat4, 'conv6_1', num_filters=512, kernel_size=(3,3,3), training=training)
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=512, kernel_size=(3,3,3), training=training)

    upconv3 = layers.deconv3D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=256, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=4, name='concat3')

    conv7_1 = layers.conv3D_layer_bn(concat3, 'conv7_1', num_filters=256, kernel_size=(3,3,3), training=training)
    conv7_2 = layers.conv3D_layer_bn(conv7_1, 'conv7_2', num_filters=256, kernel_size=(3,3,3), training=training)

    upconv2 = layers.deconv3D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=128, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=4, name='concat2')

    conv8_1 = layers.conv3D_layer_bn(concat2, 'conv8_1', num_filters=128, kernel_size=(3,3,1), training=training)
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=128, kernel_size=(3,3,1), training=training)

    upconv1 = layers.deconv3D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=64, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=4, name='concat1')

    conv9_1 = layers.conv3D_layer_bn(concat1, 'conv9_1', num_filters=64, kernel_size=(3,3,1), training=training)
    conv9_2 = layers.conv3D_layer_bn(conv9_1, 'conv9_2', num_filters=64, kernel_size=(3,3,1), training=training)

    pred = layers.conv3D_layer_bn(conv9_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1,1), activation=layers.no_activation, training=training)

    return pred


def unet_bn_2D3D_half(images, training):

    conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=64//2, kernel_size=(3,3,1), training=training)
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64//2, kernel_size=(3,3,1), training=training)

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=128//4, kernel_size=(3,3,1), training=training)
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128//2, kernel_size=(3,3,1), training=training)

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,1), strides=(2,2,1))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=256//4, kernel_size=(3,3,3), training=training)
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256//2, kernel_size=(3,3,3), training=training)

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=512//4, kernel_size=(3,3,3), training=training)
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512//2, kernel_size=(3,3,3), training=training)

    pool4 = layers.max_pool_layer3d(conv4_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv5_1 = layers.conv3D_layer_bn(pool4, 'conv5_1', num_filters=1024//2, kernel_size=(3,3,3), training=training)
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=1024//2, kernel_size=(3,3,3), training=training)

    upconv4 = layers.deconv3D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=512//2, training=training)
    concat4 = tf.concat([conv4_2, upconv4], axis=4, name='concat4')

    conv6_1 = layers.conv3D_layer_bn(concat4, 'conv6_1', num_filters=512//4, kernel_size=(3,3,3), training=training)
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=512//4, kernel_size=(3,3,3), training=training)

    upconv3 = layers.deconv3D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=256//2, training=training)
    concat3 = tf.concat([conv3_2, upconv3], axis=4, name='concat3')

    conv7_1 = layers.conv3D_layer_bn(concat3, 'conv7_1', num_filters=256//4, kernel_size=(3,3,3), training=training)
    conv7_2 = layers.conv3D_layer_bn(conv7_1, 'conv7_2', num_filters=256//4, kernel_size=(3,3,3), training=training)

    upconv2 = layers.deconv3D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=128//2, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=4, name='concat2')

    conv8_1 = layers.conv3D_layer_bn(concat2, 'conv8_1', num_filters=128//4, kernel_size=(3,3,1), training=training)
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=128//4, kernel_size=(3,3,1), training=training)

    upconv1 = layers.deconv3D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4, 1), strides=(2, 2, 1), num_filters=64//2, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=4, name='concat1')

    conv9_1 = layers.conv3D_layer_bn(concat1, 'conv9_1', num_filters=64//2, kernel_size=(3,3,1), training=training)
    conv9_2 = layers.conv3D_layer_bn(conv9_1, 'conv9_2', num_filters=64//2, kernel_size=(3,3,1), training=training)

    pred = layers.conv3D_layer_bn(conv9_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1,1), activation=layers.no_activation, training=training)

    return pred


def unet3D_bn(images, training):

    conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=32, kernel_size=(3,3,3), training=training)
    conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=64, kernel_size=(3,3,3), training=training)

    pool1 = layers.max_pool_layer3d(conv1_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=64, kernel_size=(3,3,3), training=training)
    conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=128, kernel_size=(3,3,3), training=training)

    pool2 = layers.max_pool_layer3d(conv2_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=128, kernel_size=(3,3,3), training=training)
    conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=256, kernel_size=(3,3,3), training=training)

    pool3 = layers.max_pool_layer3d(conv3_2, kernel_size=(2,2,2), strides=(2,2,2))

    conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=256, kernel_size=(3,3,3), training=training)
    conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=512, kernel_size=(3,3,3), training=training)

    upconv3 = layers.deconv3D_layer_bn(conv4_2, name='upconv3', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=512, training=training)
    concat3= tf.concat([conv3_2, upconv3], axis=4, name='concat3')

    conv5_1 = layers.conv3D_layer_bn(concat3, 'conv5_1', num_filters=256, kernel_size=(3,3,3), training=training)
    conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=256, kernel_size=(3,3,3), training=training)

    upconv2 = layers.deconv3D_layer_bn(conv5_2, name='upconv2', kernel_size=(4, 4, 4), strides=(2, 2, 2), num_filters=256, training=training)
    concat2 = tf.concat([conv2_2, upconv2], axis=4, name='concat2')

    conv6_1 = layers.conv3D_layer_bn(concat2, 'conv6_1', num_filters=128, kernel_size=(3,3,3), training=training)
    conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=128, kernel_size=(3,3,3), training=training)

    upconv1 = layers.deconv3D_layer_bn(conv6_2, name='upconv1', kernel_size=(4, 4, 2), strides=(2, 2, 2), num_filters=128, training=training)
    concat1 = tf.concat([conv1_2, upconv1], axis=4, name='concat1')

    conv8_1 = layers.conv3D_layer_bn(concat1, 'conv8_1', num_filters=64, kernel_size=(3,3,3), training=training)
    conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=64, kernel_size=(3,3,3), training=training)

    pred = layers.conv3D_layer_bn(conv8_2, 'pred', num_filters=NUM_CLASSES, kernel_size=(1,1,1), activation=layers.no_activation, training=training)

    return pred