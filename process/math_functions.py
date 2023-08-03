import math

import numpy as np
import scipy.sparse.linalg
from typing import Tuple
from sklearn.decomposition import PCA

from tqdm import tqdm

def rpca(image_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Robust Principle Component Analysis on image_mat.
    :param image_mat: The 2d flattened image matrix
    :returns: sparse parts, low rank matrices of image_mat in that order
    """

    print(f"Beginning Robust PCA on matrix of size {image_mat.shape}.")
    m = image_mat.shape[0]
    lam = 1 / math.sqrt(m)
    tol = 1e-7
    max_iter = 40
    norm_two = np.linalg.norm(image_mat, 2)
    norm_inf = np.linalg.norm(image_mat.flatten(), np.inf) / lam
    norm_fro = np.linalg.norm(image_mat, 'fro')
    dual_norm = max(norm_two, norm_inf)
    y = image_mat / dual_norm

    a_hat = np.empty(image_mat.shape)
    e_hat = np.empty(image_mat.shape)
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5

    for i in (pbar := tqdm(range(max_iter))):
        pbar.set_description(f"Iteration {i} of {max_iter}")
        temp = image_mat - a_hat + 1 / mu * y
        e_hat = (np.maximum(temp - lam / mu, 0) +
                 np.minimum(temp + lam / mu, 0))

        u, sigma, vt = (scipy.sparse.linalg.svds(
            np.asarray(image_mat - e_hat + 1 / mu * y),
            min(6, image_mat.shape[1]-2)))
        svp = (sigma > 1 / mu).sum()

        a_hat = (u[:, -svp:].dot(np.diag(sigma[-svp:] - 1 / mu)).dot(
            vt[-svp:, :]))

        z = image_mat - a_hat - e_hat
        y += mu * z
        mu = min(mu * rho, mu_bar)

        if (np.linalg.norm(z, 'fro') / norm_fro) < tol:
            break

    return a_hat, e_hat

def pca(image_mat: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs Principle Component Analysis on image_mat.
    :param image_mat: The 2d flattened image matrix. Each column represents a different instance. 
    :param k: The number of principal components desired
    :returns: pca matrix, mean of rows
    """
    
    # Standardize the data row-wise. This finds the average of each feature (pixel of the image) and std

    mean = image_mat.mean(axis=1)
    print(mean.shape)
    std =  image_mat.std(axis=1)
    standardized = np.empty(image_mat.shape)

    print("Beginning PCA standardization")
    for i in range(image_mat.shape[1]):

        standardized[:,i] = (image_mat[:,i] - mean)/std
    print("PCA standardization complete\n")

    # Get the eigenvalues and eigenvectors using the fast method

    small_matrix = standardized.T @ standardized

    eigenvalues_small, eigenvectors_small = np.linalg.eig(small_matrix)
    order = np.argsort(eigenvalues_small[::-1])
    eigenvectors_small_sorted = eigenvectors_small[:,order]

    eigenvectors_sorted = standardized @ eigenvectors_small_sorted

    # Returns the index order of sorting from max eigenvalue to min, for dimensional reduction.
    
    return eigenvectors_sorted[:,:k], mean, std
    

def fft2_series(img_mat: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    For each column in img_mat, img_mat[:, i] the fft2 modes are extracted and
    placed into the corresponding column of the returned matrix.

    :param img_mat: Matrix to process.
    :param shape: Original (width, height) of each column of img_mat.
    :returns: New numpy.ndarray matrix, where return[:, i] is the fft2 modes of
        img_mat[:, i].
    """
    fourier = np.empty(img_mat.shape)

    # column wise fourier transform. Each flattened image is transformed.
    for i in range(img_mat.shape[1]):
        fourier[:, i] = (
            np.abs(np.fft.fft2(img_mat[:, i].reshape(shape))).flatten())
    return fourier
