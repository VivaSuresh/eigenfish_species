from pathlib import Path
from typing import List, Tuple

import cv2
import imageio
import numpy as np


def load_img_mat(files: List[Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Loads all files as images in black and white, flattens them and returns them
    as a numpy.ndarray.

    Loads all files as RGB images, separates each channel, flattens, and returns 3 
    np.ndarrays, one for each channel

    :param files: List of image file paths to load. All should be of the same
        resolution.
    :return: 3 Numpy.ndarray matrices with each row as flattened images.
    """

    shape = imageio.imread(files[0].as_posix()).shape
    # Input must be 3 channel RGB or BGR
    assert(len(shape) == 3)
    (len(files), shape[0] * shape[1])
    img_mat_r = np.empty((len(files), shape[0] * shape[1]))
    img_mat_g = np.empty((len(files), shape[0] * shape[1]))
    img_mat_b = np.empty((len(files), shape[0] * shape[1]))

    # Commencing loading the matrices
    for i, file in enumerate(files):

        image = cv2.imread(file.as_posix())
        r, g, b = cv2.split(image)

        img_mat_r[i,:] = r.flatten()
        img_mat_g[i,:] = g.flatten()
        img_mat_b[i,:] = b.flatten()
    
    # Combine into a 3d matrix for returning
    img_mat_rgb = np.dstack((r,g,b))

    return img_mat_rgb, shape

