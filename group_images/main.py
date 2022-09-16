# Created by Carlos Ramirez at 16/09/2022
import os
import shutil
from typing import Optional

from .feature_extractor import FeatureExtractor
from .separate import Separate


def save_images(result: dict, out_dir: str, zfill: Optional[int] = None) -> None:
    """
    Save result from clustering to the output dictionaru
    :param result: return value from class Separate.cluster_images
    :param out_dir: output directory to save images
    :param zfill: optional parameters to add left zeros to the output cluster directories
    :return: None
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    for image_path, cluster_id in result.items():
        dir_name = str(cluster_id) if zfill is None else str(cluster_id).zfill(zfill)
        dir_path = os.path.join(out_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        shutil.copy2(image_path, dir_path)


def cluster_images(input_dir: str, output_dir: str, min_cluster, max_cluster: int,
                   model: str = 'resnet50') -> None:
    """
    Easy mode to cluster images with less parameters, for a more complete experience
    consider using FeatureExtractor and Separate by yourself.
    :param input_dir: directory containing all the input images
    :param output_dir: directory where the clusters will be saved
    :param min_cluster: minimum number of clusters to have
    :param max_cluster: maximum number of cluster to have
    :param model: model to use for feature extraction
    :return: None
    """
    extractor = FeatureExtractor(input_dir, model)
    dict_features = extractor.get_features(batch_size=8)
    cluster = Separate(dict_features, min_cluster, max_cluster)
    result = cluster.cluster_images(100, early_stop_inertia=5000)
    save_images(result, output_dir)
