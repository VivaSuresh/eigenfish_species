from .math_functions import fft2_series, rpca, pca
import numpy as np
from typing import Tuple
from pathlib import Path
from util import display_img


class Processor:
    def process(self, img_mat: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """
        Process img_mat to prepare it for training/classification.

        :param img_mat: Matrix with each column a flattened image.
        :param shape: Original (width, height) of each image.
        """

        components, mean, std = pca(img_mat, k=21)
            
        # img = np.reshape(robust[:,0], shape)
        # lowrank = np.reshape(robust[1][:,0], shape)

        # print("lowrank shape: ", lowrank.shape)

        # display_img(img)
        # display_img(sparse, normalize=True)
        # for i in range(21):
            
        #     img = np.reshape(components[:,i]*std+mean, shape)
        #     display_img(img, normalize=True)
            

        return components#fft2_series(rpca[1], shape)
