import numpy as np
import tensorflow as tf


class CycleGAN:
    max_summary_images = 2
    c_times = 5

    def __init__(self, translators, datasets):
        self.translators = translators
        self.datasets = datasets
        assert len(self.translators) == len(self.datasets)

    def __call__(self, batch_size, steps, model_path):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter(model_path, sess.graph)
            tf.global_variables_initializer().run()

            from time import time
            start_time = time()
            for step in range(steps):
                import sys
                print(step, end=" ")
                sys.stdout.flush()

                if step < 25:
                    CycleGAN.c_times = 100
                else:
                    CycleGAN.c_times = 10

                for i, (translator, dataset) in enumerate(zip(self.translators, self.datasets)):
                    for _ in range(CycleGAN.c_times):
                        eta = np.random.rand(batch_size, 1, 1, 1)  # sampling from uniform distribution
                        this_real_image = dataset.next_batch_real(batch_size)
                        other_real_image = self.datasets[not i].next_batch_real(batch_size)

                        sess.run(translator.c_optimizer,
                                 feed_dict={translator.real_image_this: this_real_image,
                                            translator.real_image_other: other_real_image,
                                            translator.eta: eta})

                    this_real_image = dataset.next_batch_real(batch_size)
                    sess.run(translator.g_optimizer, feed_dict={translator.real_image_this: this_real_image})

                if step % 100 == 0:
                    for i, (translator, dataset) in enumerate(zip(self.translators, self.datasets)):
                        eta = np.random.rand(batch_size, 1, 1, 1)  # sampling from uniform distribution
                        this_real_image = dataset.next_batch_real(batch_size)
                        other_real_image = self.datasets[not i].next_batch_real(batch_size)

                        translator_summary = sess.run(translator.summaries,
                                                      feed_dict={translator.real_image_this: this_real_image,
                                                                 translator.real_image_other: other_real_image,
                                                                 translator.eta: eta})
                        writer.add_summary(translator_summary, step)
                    print("\rSummaries generated! Step", step, " Time == %.2fs" % (time() - start_time))
                    start_time = time()
