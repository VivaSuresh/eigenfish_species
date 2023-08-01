from pathlib import Path
from typing import List, Tuple

import cv2
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

    shape = (128,128)

    img_mat_r = np.empty((shape[0] * shape[1], len(files)))
    img_mat_g = np.empty((shape[0] * shape[1], len(files)))
    img_mat_b = np.empty((shape[0] * shape[1], len(files)))

    # Commencing loading the matrices
    for i, file in enumerate(files):

        image = cv2.imread(file.as_posix())

        # Input must be 3 channel RGB or BGR

        assert(len(image.shape) == 3)

        b, g, r = cv2.split(image)
        
        b = cv2.resize(b, shape)
        g = cv2.resize(g, shape)
        r = cv2.resize(r, shape)

        img_mat_r[:,i] = r.flatten()
        img_mat_g[:,i] = g.flatten()
        img_mat_b[:,i] = b.flatten()
    
    # Combine into a 3d matrix for returning
    img_mat_bgr = np.dstack((img_mat_b,img_mat_g,img_mat_r))

    return img_mat_bgr, shape


def display_img(img: np.ndarray, add_noise=False, normalize=False):
    """
    Displays a given matrix with or without noise based on user input

    :param img: The img matrix to be displayed
    :param add_noise: Whether or not to add normal noise. Default is false
    :param normalize: Whether or not to normalize the image
    """    

    if normalize:
        cv2.normalize(img, None, 0, 255)

    if add_noise:

        # Random gaussian noise with a reach of 10
        gaussian = np.abs(np.random.normal(0, 10, (img.shape[0],img.shape[1])))
            
        noisy_img = (img+gaussian)
        noisy_img[noisy_img>255] = 255
        img = noisy_img.astype(np.uint8)

    else:

        img = img.astype(np.uint8)
    
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
