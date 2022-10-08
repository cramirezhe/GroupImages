# Created by Carlos Ramirez at 16/09/2022
import configparser
import logging
import os
import shutil
from typing import List, Optional

from .feature_extractor import FeatureExtractor
from .separate import Separate


def save_images(result: dict, out_dir: str, zfill: Optional[int] = None) -> None:
    """
    Save result from clustering to the output dictionaru
    Args:
        result (dict): return value from class Separate.cluster_images
        out_dir (str): output directory to save images
        zfill (Optional[int]): optional parameters to add left zeros to the output cluster
                             directories

    Returns:
        None
    """
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    for image_path, cluster_id in result.items():
        dir_name = str(cluster_id) if zfill is None else str(cluster_id).zfill(zfill)
        dir_path = os.path.join(out_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        shutil.copy2(image_path, dir_path)


def cluster_images(params: dict) -> None:
    """
    Easy mode to cluster images with less parameters, for a more complete experience
    consider using FeatureExtractor and Separate by yourself.
    Args:
        params (dict): dictionary including the following fields:
                       input_dir [str]: path to the directory containing the images.
                       output_dir [str]: destination directory where the images will be stored.
                       pooling [str or None] (optional): pooling for the feature extractor
                                                         [avg or max] (by default avg) set it
                                                         to None to remove pooling layer.
                       batch_size [int] (optional): size of the input batch for
                                                    the feature extractor (by default 8).
                       min_cluster [int]: minimum number of cluster we are looking
                                          for (must be > 1).
                       max_cluster [int]: maximum number of cluster we are looking for
                       (must be < than the # of images in input_dir.
                       iterations [int] (optional): number of iterations to run our
                                                    kmeans algorithm.
                       random_state [int] (optional): initial random state, use it to repeat
                                                      results.
                       early_stop_inertia [float] (optional): in kmeans inertia is bellow
                                                              this value we will stop the algorithm.

    Returns:
        None
    """
    extractor = FeatureExtractor(params['input_dir'], params['output_dir'],
                                 params.get('pooling', 'avg'))
    dict_features = extractor.get_features(batch_size=params.get('batch_size', 8))
    cluster = Separate(dict_features, params['min_cluster'], params['max_cluster'])
    result = cluster.cluster_images(
        iterations=params.get('iterations', 100),
        random_state=params.get('random_state', None),
        early_stop_inertia=params.get('stop_inertia', None)
    )
    zfill = params.get('zfill', None)
    save_images(result, params['output_dir'], None if zfill is None else int(zfill))


def verify_ini_sections(config: configparser.ConfigParser,  sections: List[str]) -> bool:
    """Verifies if a ini file has all the necessary sections."""
    for section in sections:
        if section not in config.sections():
            logging.error(f"{section} is not part of {config.sections()} in config file.")
            return False
    return True


def validate_section_value(config: configparser.ConfigParser, section: str,
                           key: str, msg: str) -> str:
    """
    Validates that a configuration file in ini format has a given field in a given section.
    Args:
        config (ConfigParser): parsed configuration file
        section (str): section to analyse
        key (str): field in the section we are looking for
        msg (str): error message

    Returns:
        str: the value in the configuration file in string format
    Raises:
        ValueError: if the field is not found in the section
    """
    val = config[section].get(key, None)
    if val is None:
        raise ValueError(msg)
    return str(val)


def parse_conf_file(path: str) -> dict:
    """Parses a configuration file in ini format"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} does not exists")
    sections = ['GENERAL', 'FEAT_EXT', 'CLUSTER']
    config = configparser.ConfigParser(inline_comment_prefixes=';')
    config.read(path)
    if not verify_ini_sections(config, sections):
        raise ValueError
    params = {
        'input_dir': validate_section_value(
            config,
            'GENERAL', 'INPUT_PATH',
            "INPUT_PATH was not found in GENERAL section"
        ),
        'output_dir': validate_section_value(
            config,
            'GENERAL', 'OUTPUT_PATH',
            "OUTPUT_PATH was not found in GENERAL section"
        ),
        'model': validate_section_value(
            config,
            'FEAT_EXT', 'MODEL',
            "MODEL was not found in FEAT_EXT section"),
        'pooling': config['FEAT_EXT'].get('POOLING', None),
        'batch_size': int(config['FEAT_EXT'].get('BATCH_SIZE', '4')),
        'min_cluster': int(validate_section_value(
            config,
            'CLUSTER',
            'MIN_CLUSTER',
            "MIN_CLUSTER was not found in CLUSTER section"
        )),
        'max_cluster': int(validate_section_value(
            config,
            'CLUSTER',
            'MAX_CLUSTER',
            "MAX_CLUSTER was not found in CLUSTER section"
        )),
        'iterations': int(config['CLUSTER'].get('ITERATIONS', '100')),
    }
    params['random_state'] = config['CLUSTER'].get('RANDOM_STATE', None)
    if params['random_state'] is not None:
        params['random_state'] = int(params['random_state'])
    params['stop_inertia'] = config['CLUSTER'].get('STOP_INERTIA', None)
    if params['stop_inertia'] is not None:
        params['stop_inertia'] = float(params['stop_inertia'])
    return params


def run_by_conf_file(path: str) -> None:
    """Cluster images using a configuration file as input."""
    params = parse_conf_file(path)
    cluster_images(params)
