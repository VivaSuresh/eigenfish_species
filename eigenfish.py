from classify.classify import Classifier
from process.process import Processor
import cv2
from typing import Tuple, List
import numpy as np
from util import display_img


class Eigenfish:
    def __init__(self, shape: Tuple[int, int], training_file=None, processor=None,
                 classifier=None):
        """
        Initialize Eigenfish instance.

        :param shape: (Width, Height) of each images
        :param training_file: If not None, saved training data from previous
            instance, else new model created.
        :param processor: If not None, custom Processor, else default.
        :param classifier: If not None, custom Classifier, else default.
        """
        self.shape = shape
        self.processor = Processor() if processor is None else processor()
        self.classifier = Classifier() if classifier is None else classifier()
        if training_file is not None:
            self.load(training_file)

    def train(self, img_mat: np.ndarray, label_arr: List[str]):
        """
        Helper to add to current model's training.

        :param img_mat: Column-wise matrix of flattened images.
        :param label_arr: List of labels, where label_arr[i] corresponds to
            img_mat[:, i].
        """

        return self.processor.process(img_mat, self.shape)
        #  

    def train_rgb(self, img_mat_rgb: np.ndarray, label_arr: List[str]):
        """
        Add to current model's training.

        :param img_mat_rgb: 3 matrices with column-wise matrix of flattened images.
        :param label_arr: List of labels, where label_arr[i] corresponds to
            img_mat[:, i].
        """
        print(img_mat_rgb.shape)
        b_mat, g_mat, r_mat = cv2.split(img_mat_rgb)
        
        r = self.train(r_mat, label_arr)
        g = self.train(g_mat, label_arr)
        b = self.train(b_mat, label_arr)
        
        for i in range(21):

            b_shaped = np.reshape(b[:,i], self.shape)
            g_shaped = np.reshape(g[:,i], self.shape)
            r_shaped = np.reshape(r[:,i], self.shape)

            img_mat_bgr = np.dstack((b_shaped,g_shaped,r_shaped))

            print(np.amax(img_mat_bgr))
            display_img(img_mat_bgr, normalize=True)



    def classify(self, img_mat: np.ndarray):
        """
        Classify img_mat based on current training.

        :param img_mat: Column-wise matrix of flattened images.
        :return: List of labels, one for each column of img_mat.
        """
        return self.classifier.classify(self.processor.process(img_mat,
                                                               self.shape))

    def cross_validate(self, img_mat: np.ndarray, label_arr: List[str]):
        """
        Cross-validates the trained model. Img_mat will be run through the
        classifier, and each predicted label of img_mat[:, i] compared with
        label_arr[i]. The percent same is returned.

        :param img_mat: Column-wise matrix of flattened images.
        :param label_arr: List of labels, where label_arr[i] corresponds to
            img_mat[:, i].
        :return: Percent of labels that are the same.
        """
        return self.classifier.cross_validate(
            self.processor.process(img_mat, self.shape), label_arr)

    def load(self, filename: str):
        """
        Loads saved training data and overwrites current model. Use only on data
        you have previously saved, and make sure to use the same processor and
        classifier.

        :param filename: File to load into classifier.
        """
        self.classifier.load(filename)

    def save(self, filename: str):
        """
        Saves currently trained model to filename.

        :param filename: File to save from classifier.
        """
        self.classifier.save(filename)
