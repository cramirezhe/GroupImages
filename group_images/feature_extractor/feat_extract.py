# Created by Carlos Ramirez at 09/09/2022
import logging
import os.path
import subprocess
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.applications import (densenet, efficientnet_v2,
                                           inception_v3, mobilenet_v2,
                                           resnet_v2)

from .image_loader import ImageLoader


class FeatureExtractor:
    """Class used to extract features from a directory"""
    models = {
        'densenet121': [densenet.DenseNet121,
                        densenet.preprocess_input,
                        (224, 224, 3)],
        'densenet169': [densenet.DenseNet169,
                        densenet.preprocess_input,
                        (224, 224, 3)],
        'densenet201': [densenet.DenseNet201,
                        densenet.preprocess_input,
                        (224, 224, 3)],
        'efficientnetv2_b0': [efficientnet_v2.EfficientNetV2B0,
                              efficientnet_v2.preprocess_input,
                              (224, 224, 3)],
        'efficientnetv2_b1': [efficientnet_v2.EfficientNetV2B1,
                              efficientnet_v2.preprocess_input,
                              (240, 240, 3)],
        'efficientnetv2_b2': [efficientnet_v2.EfficientNetV2B2,
                              efficientnet_v2.preprocess_input,
                              (260, 260, 3)],
        'inceptionv3': [inception_v3.InceptionV3,
                        inception_v3.preprocess_input,
                        (299, 299, 3)],
        'mobilenet_v2': [mobilenet_v2.MobileNetV2,
                         mobilenet_v2.preprocess_input,
                         (224, 224, 3)],
        'resnet50': [resnet_v2.ResNet50V2,
                     resnet_v2.preprocess_input,
                     (224, 224, 3)],
        'resnet101': [resnet_v2.ResNet101V2,
                      resnet_v2.preprocess_input,
                      (224, 224, 3)],
        'resnet152': [resnet_v2.ResNet152V2,
                      resnet_v2.preprocess_input,
                      (224, 224, 3)],
    }
    """Dictionary used to select models and preprocessing functions"""

    def __init__(self, dir_path: str, model: str = 'resnet50',
                 pooling: Optional[str] = 'avg'):
        """
        Initialize the model feature extractor and image loader.
        Args:
            dir_path: path where our unsorted images are located.
            model: model to be use as feature extractor, by default, we will use resnet50.
            pooling: Pooling technique for the neural network default is None, this means that
                     we will use the 4D output vector as features. Otherwise you can select
                     between 'avg' and 'max' pooling.
        """
        self._models = FeatureExtractor.models
        # Verify input path
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"{dir_path} is not a valid directory.")
        self._input_dir = dir_path
        # Clean input parameters
        model = model.casefold()
        if pooling not in ['avg', 'max']:
            # Supported modes are avg and max. Otherwise, None is selected
            pooling = 'avg'
        # Init model
        self._load_model(model=model, pooling=pooling)
        # Create ImageLoader
        self._images_path = self.find_images_dir()
        self._data_loader = ImageLoader(self._images_path, self._input_size[:2],
                                        self._preprocess_fnc)
        self._loader = tf.data.Dataset.from_generator(
            self._data_loader,
            output_types=(tf.float32, tf.string)
        )

    def get_features(self, batch_size: int = 8) -> dict:
        """
        Extract features from the images defined in the constructor.
        Args:
            batch_size (int): batch size to use for image processing

        Returns:
            dict: a dictionary using the image path as key and the feature vector as content.
        """
        bs = self._loader.batch(batch_size)
        dict_imgs = {}
        for batch in bs:
            input_batch, imgs_paths = batch
            features = self._net.predict(input_batch)
            # Add the features to our dictionary
            for i in range(input_batch.shape[0]):
                img_path = imgs_paths[i].numpy().decode('utf-8')
                dict_imgs[img_path] = features[i].flatten()
        return dict_imgs

    def update_input_dir(self, path: str) -> bool:
        """
        Updates the input directory.
        Args:
            path: a valid path to a new directory

        Returns:
            bool: True if the update was successful and False otherwise.
        """
        if not os.path.isdir(path):
            logging.warning(f"{path} is not a valid directory, ignoring...")
            return False
        images_paths = self.find_images_dir(path)
        if len(images_paths) <= 0:
            logging.warning(f"{path} does not contain images, ignoring...")
            return False
        # We did find images, so update it
        logging.info("Updating images")
        self._input_dir = path
        self._images_path = images_paths
        return True

    def get_input_dir(self) -> str:
        """Returns current input directory"""
        return self._input_dir

    def find_images_dir(self, path: Optional[str] = None) -> List[str]:
        """
        Find recursively all the images in the input directory from constructor
        or another path defined by the input parameter of this function
        Args:
            path: path to search images or None if we want to use class directory path

        Returns:
            List[str]: A list of images paths.
        """
        input_dir = self._input_dir if path is None else path
        find_cmd = f"find {input_dir} -type f -exec file --mime-type {{}} \\+ "
        find_cmd += "| awk -F: '{{if ($2 ~/image\\//) print $1}}'"
        try:
            images =\
                subprocess.run(find_cmd, capture_output=True, shell=True).stdout.decode('utf-8')
        except subprocess.SubprocessError:
            logging.error(f"[Error] Failed to search images in {input_dir}")
            raise subprocess.SubprocessError
        # Remove final empty line
        list_images = images.split('\n')[:-1]
        return list_images

    def print_model(self):
        """Print current model architecture"""
        print(self._net.summary())

    def verify_model(self):
        """Verifies if model was created correctly"""
        return self._net is not None

    def _load_model(self, model: str = 'resnet50', pooling: Optional[str] = None):
        """Load the selected model and preprocessing function."""
        model_builder, self._preprocess_fnc, in_size = self._models.get(model, [None, None, None])
        if model_builder is None or self._preprocess_fnc is None:
            # Impossible to get a model from dictionary, return a default model
            model_builder, self._preprocess_fnc, in_size = self._models['resnet50']
        self._input_size = in_size
        self._net = model_builder(include_top=False, input_shape=self._input_size,
                                  weights='imagenet', pooling=pooling)
