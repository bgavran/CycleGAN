import tensorflow as tf


class FCGenerator:
    def __init__(self, name="Generator", img_size=256):
        self.name = name
        self.img_size = img_size
        self.channels = 3

    def __call__(self, image, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            image = tf.reshape(image, [-1, self.img_size * self.img_size * self.channels])

            image = tf.layers.dense(image, 512, activation=tf.nn.relu)
            image = tf.layers.dense(image, 512, activation=tf.nn.relu)
            image = tf.layers.dense(image, 512, activation=tf.nn.relu)
            image = tf.layers.dense(image, self.img_size * self.img_size * self.channels, activation=tf.nn.sigmoid)

            image = tf.reshape(image, [-1, self.img_size, self.img_size, self.channels])
            return image


class ConvGenerator:
    def __init__(self, name="Generator", img_size=256):
        self.name = name
        self.img_size = img_size

    def __call__(self, image, reuse=None):
        with tf.variable_scope(self.name, reuse=reuse):
            act = tf.nn.relu
            kwargs_downsample = {"kernel_size": (4, 4), "strides": (4, 4), "padding": "valid"}

            # image is 256x256x3
            image = tf.layers.conv2d(image, filters=128, **kwargs_downsample, activation=act)

            # image is 64x64x128
            image = tf.layers.conv2d(image, filters=256, **kwargs_downsample, activation=act)

            # image is 16x16x256
            image = tf.layers.conv2d(image, filters=512, **kwargs_downsample, activation=act)

            # -------------- image is 4x4x512
            pad = [[0, 0], [2, 2], [2, 2], [0, 0]]
            kwargs_upsample = {"kernel_size": (5, 5), "strides": (1, 1), "padding": "valid"}
            res_met = tf.image.ResizeMethod.NEAREST_NEIGHBOR

            image = tf.pad(image, pad, mode="SYMMETRIC")
            image = tf.layers.conv2d(image, filters=512, **kwargs_upsample, activation=act)
            image = tf.image.resize_images(image, (16, 16), method=res_met)

            # image is 16x16x256

            image = tf.pad(image, pad, mode="SYMMETRIC")
            image = tf.layers.conv2d(image, filters=256, **kwargs_upsample, activation=act)
            image = tf.image.resize_images(image, (64, 64), method=res_met)

            # image is 64x64x128

            image = tf.pad(image, pad, mode="SYMMETRIC")
            image = tf.layers.conv2d(image, filters=128, **kwargs_upsample, activation=act)
            image = tf.image.resize_images(image, (256, 256), method=res_met)

            # image is 256x256x128

            image = tf.pad(image, pad, mode="SYMMETRIC")
            image = tf.layers.conv2d(image, filters=3, activation=tf.nn.sigmoid, **kwargs_upsample)

            # image is 256x256x3

            return image
