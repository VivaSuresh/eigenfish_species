from .math import fft2_series, rpca
import numpy as np
from typing import Tuple
from pathlib import Path
from util import display_img


class Processor:
    def process(self, img_mat: np.ndarray, shape: Tuple[int, int]):
        """
        Process img_mat to prepare it for training/classification.

        :param img_mat: Matrix with each column a flattened image.
        :param shape: Original (width, height) of each image.
        """

        robust = rpca(img_mat)

        sparse = np.reshape(robust[0][:,0], shape)
        lowrank = np.reshape(robust[1][:,0], shape)

        print("lowrank shape: ", lowrank.shape)

        display_img(lowrank, normalize=True)
        display_img(sparse, normalize=True)

        return fft2_series(rpca[1], shape)
