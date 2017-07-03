import tensorflow as tf


class Translator:
    max_summary_images = 2

    # Cycle consistency loss parameter
    lambd = 1

    # Gradient penalty parameter
    gp_lambda = 10

    def __init__(self, encoder,
                 decoder,
                 critic,
                 name="Translator",
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.5, beta2=0.9),
                 reuse=None):
        self.encoder = encoder
        self.decoder = decoder
        self.critic = critic
        self.name = name
        self.optimizer = optimizer

        assert encoder.img_size == decoder.img_size == critic.img_size

        self.img_size = encoder.img_size
        self.channels = 3

        self.real_image_this = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.channels],
                                              name="ImageThis")
        self.real_image_other = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.channels],
                                               name="ImageOther")

        self.fake_image_other = self.encoder(self.real_image_this, reuse=reuse)
        self.cyclic_image = self.decoder(self.fake_image_other, reuse=reuse)

        self.cyclic_loss = tf.reduce_mean(tf.squared_difference(self.real_image_this, self.cyclic_image),
                                          name="Cyclic_loss")

        self.c_fake_image_other = tf.reduce_mean(self.critic(self.fake_image_other))
        self.c_real_image_other = tf.reduce_mean(self.critic(self.real_image_other, reuse=True))

        self.g_loss = self.c_fake_image_other + Translator.lambd * self.cyclic_loss
        self.c_loss = tf.reduce_mean(self.c_real_image_other - self.c_fake_image_other)

        # Applying the WGAN-GP
        with tf.name_scope("Gradient_penalty"):
            self.eta = tf.placeholder(tf.float32, [None, 1, 1, 1], name="Eta")
            # interpolation between real and generated
            x_interp = self.eta * self.real_image_other + (1 - self.eta) * self.fake_image_other

            # gradient of interpolated
            grads = tf.gradients(self.critic(x_interp, reuse=True), x_interp)[0]

            # L2 norm of gradients, shape of [batch_size]
            grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))

            difference_from_1 = tf.squared_difference(grad_norm, 1)
            self.c_loss += Translator.gp_lambda * tf.reduce_mean(difference_from_1)

        self.c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.critic.name)
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.encoder.name) + \
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.decoder.name)

        self.g_optimizer = self.optimizer.minimize(self.g_loss, var_list=self.g_vars,
                                                   name=self.name + "_Optimizer")
        self.c_optimizer = self.optimizer.minimize(self.c_loss, var_list=self.c_vars,
                                                   name=self.critic.name + "_Optimizer")

        real_img_this = tf.summary.image("Real image this", self.real_image_this,
                                         max_outputs=Translator.max_summary_images)
        real_img_other = tf.summary.image("Real image other", self.real_image_other,
                                          max_outputs=Translator.max_summary_images)
        fake_image_other = tf.summary.image("Generated image", self.fake_image_other,
                                            max_outputs=Translator.max_summary_images)
        cyclic_image = tf.summary.image("Cyclic image", self.cyclic_image, max_outputs=Translator.max_summary_images)
        c_cost = tf.summary.scalar("Critic cost", self.c_loss)
        g_cost = tf.summary.scalar("Generator cost", self.g_loss)

        summary_list = [real_img_this, real_img_other, fake_image_other, cyclic_image, c_cost, g_cost]
        # Gradient summary
        from tensorflow.python.framework import ops
        for gradient, variable in self.optimizer.compute_gradients(self.c_loss):
            if gradient is not None:
                summary_list.append(tf.summary.histogram(variable.name, variable))
                summary_list.append(tf.summary.histogram(variable.name + "/gradients",
                                                         gradient.values if isinstance(gradient,
                                                                                       ops.IndexedSlices) else gradient))

        self.summaries = tf.summary.merge(summary_list)
