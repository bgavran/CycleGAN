import tensorflow as tf


class FCCritic:
    def __init__(self, name, img_size=256):
        self.name = name
        self.img_size = img_size
        self.channels = 3

    def __call__(self, image, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            image = tf.reshape(image, [-1, self.img_size * self.img_size * self.channels])

            image = tf.layers.dense(image, 512, tf.nn.relu)
            image = tf.layers.dense(image, 512, tf.nn.relu)
            image = tf.layers.dense(image, 512, tf.nn.relu)
            image = tf.layers.dense(image, 1)
            assert image.shape[1] == 1
            return image


class ConvCritic:
    def __init__(self, name="Critic", img_size=256):
        self.name = name
        self.img_size = img_size

    def __call__(self, image, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            act = tf.nn.relu
            kernel4 = (4, 4)
            kwargs_downsample = {"kernel_size": kernel4, "strides": kernel4, "padding": "valid"}

            # image is 256x256x3
            image = tf.layers.conv2d(image, filters=128, **kwargs_downsample, activation=act)

            # image is 64x64x128
            image = tf.layers.conv2d(image, filters=256, **kwargs_downsample, activation=act)

            # image is 16x16x256
            image = tf.layers.conv2d(image, filters=512, **kwargs_downsample, activation=act)

            # image is 4x4x512

            image = tf.reshape(image, [-1, 4 * 4 * 512])
            image = tf.layers.dense(image, 1)

            assert image.shape[1] == 1
            return image
