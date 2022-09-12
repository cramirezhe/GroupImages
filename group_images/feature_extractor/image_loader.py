# Created by Carlos Ramirez at 10/09/2022
from typing import Callable, List, Tuple

import cv2
import numpy as np
import tensorflow as tf


class ImageLoader:

    def __init__(self, list_paths: List[str],
                 resize_shape: Tuple[int, int],
                 preprocessing: Callable[[np.ndarray, str], np.ndarray]):
        """
        Initialized our Image Loader
        :param list_paths: a list of str of the input images path
        :param resize_shape: neural network input shape
        :param preprocessing: pre processing function for normalization
        """
        self._list_paths = list_paths
        self._preprocess_fnc = preprocessing
        self._resize_dims = resize_shape

    def __getitem__(self, idx):
        """Returns a new par input image, image path"""
        img = cv2.imread(self._list_paths[idx])
        img = cv2.resize(img, self._resize_dims, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        float_img = self._preprocess_fnc(img)
        return tf.convert_to_tensor(float_img, dtype=tf.float32), self._list_paths[idx]

    def __call__(self, *args, **kwargs):
        """Yields the batch for tensorflow inference"""
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        """Return the lenght of our dataset"""
        return len(self._list_paths)
