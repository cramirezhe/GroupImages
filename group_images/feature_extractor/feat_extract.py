# Created by Carlos Ramirez at 09/09/2022
"""Feature extractor class."""
import os.path
import subprocess
from typing import Optional

from tensorflow.keras.applications import (densenet, efficientnet_v2,
                                           inception_v3, mobilenet_v2,
                                           resnet_v2)


class FeatureExtractor:
    models = {
        'densenet121': [densenet.DenseNet121, densenet.preprocess_input],
        'densenet169': [densenet.DenseNet169, densenet.preprocess_input],
        'densenet201': [densenet.DenseNet201, densenet.preprocess_input],
        'efficientnetv2_s': [efficientnet_v2.EfficientNetV2S,
                             efficientnet_v2.preprocess_input],
        'efficientnetv2_m': [efficientnet_v2.EfficientNetV2M,
                             efficientnet_v2.preprocess_input],
        'efficientnetv2_l': [efficientnet_v2.EfficientNetV2L,
                             efficientnet_v2.preprocess_input],
        'inceptionv3': [inception_v3.InceptionV3, inception_v3.preprocess_input],
        'mobilenet_v2': [mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input],
        'resnet50': [resnet_v2.ResNet50V2, resnet_v2.preprocess_input],
        'resnet101': [resnet_v2.ResNet101V2, resnet_v2.preprocess_input],
        'resnet152': [resnet_v2.ResNet152V2, resnet_v2.preprocess_input],
    }
    """Class used to extract features from a directory"""

    def __init__(self, dir_path: str, model: str = 'resnet50', pooling: Optional[str] = None):
        """
        Initialize the model feature extractor and image loader.
        :param dir_path: path where our unsorted images are located.
        :param model: model to be use as feature extractor, options are:
                      [densenet121, densenet169, densenet201, efficientnetv2_s,
                       efficientnetv2_m, efficientnetv2_l, inceptionv3, mobilenet_v2,
                       resnet50, resnet101, resnet152].
                       By default, we will use resnet50.
        :param pooling: Pooling technique for the neural network default is None, this means that
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
            pooling = None
        # Init model
        self._load_model(model=model, pooling=pooling)

    def find_images_dir(self, path: Optional[str] = None):
        """
        Find recursively all the images in the input directory from constructor
        or another path defined by the input parameter of this function
        :param path: path to search images or None if we want to use class directory path
        :return:
            A list of images paths.
        """
        input_dir = self._input_dir if path is None else path
        find_cmd = f"find {input_dir} -type f -exec file --mime-type {{}} \\+ "
        find_cmd += "| awk -F: '{{if ($2 ~/image\\//) print $1}}'"
        try:
            images =\
                subprocess.run(find_cmd, capture_output=True, shell=True).stdout.decode('utf-8')
        except subprocess.SubprocessError:
            print(f"[Error] Failed to search images in {input_dir}")
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
        model_builder, self._preprocess_fnc = self._models.get(model, [None, None])
        if model_builder is None or self._preprocess_fnc is None:
            # Impossible to get a model from dictionary, return a default model
            model_builder, self._preprocess_fnc = self._models['resnet50']
        self._net = model_builder(include_top=False, weights='imagenet', pooling=pooling)
