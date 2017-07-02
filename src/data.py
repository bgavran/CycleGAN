import scipy.misc
import numpy as np
from utils import *

project_path = ProjectPath("log")


class Data:
    """
    Abstract class that CycleGAN uses
    """

    def next_batch_real(self, batch_size):
        """

        :param batch_size:
        :return: Tensor of real images in the shape [batch_size, height, width, channels]
        """
        raise NotImplementedError()

    @staticmethod
    def read_image(path):
        # dividing with 256 because we need to get it in the [0, 1] range
        return scipy.misc.imread(path).astype(np.float) / 256


class Animal:
    def __init__(self, animal_path, img_size=256):
        self.default_img_size = 256
        self.img_size = img_size
        self.animal_folder_path = os.path.join(project_path.data_path, "horse2zebra", animal_path)

        self.animal_path = []
        for (dirpath, dirnames, fnames) in os.walk(self.animal_folder_path):
            for fname in fnames:
                self.animal_path.append(os.path.join(dirpath, fname))

        # for now only
        self.animal_path = self.animal_path[:100]

        self.images = np.zeros((len(self.animal_path), self.img_size, self.img_size, 3))
        self.num_examples = len(self.animal_path)

        for i, img_path in enumerate(self.animal_path):
            if i % 100 == 1:
                print(i)
            self.images[i] = self.get_image(img_path, resize_dim=self.img_size)

    def get_image(self, path, resize_dim=None):
        img = Data.read_image(path)
        if resize_dim is not None:
            rev_rat = self.default_img_size / resize_dim  # ratio
            assert rev_rat.is_integer()
            rev_rat = int(rev_rat)
            img = img[::rev_rat, ::rev_rat]
        return img

    def next_batch_real(self, batch_size):
        locations = np.random.randint(0, self.num_examples, batch_size)
        return self.images[locations, ...]
