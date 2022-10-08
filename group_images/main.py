# Created by Carlos Ramirez at 19/09/2022
import argparse
import logging

from . import __version__
from .interface_file import cluster_images, run_by_conf_file


def terminal_exec():
    logging.getLogger().setLevel(logging.INFO)
    ap = argparse.ArgumentParser(description="Cluster images in a given directory")
    ap.add_argument('-m', '--mode', required=True, type=int,
                    help="Select mode to use: 0= by configuration file, 1= by args parameters")
    ap.add_argument('-c', '--configuration', type=str, required=False, default=None,
                    help="Path to the configuration file")
    ap.add_argument('-i', '--input-dir', type=str, required=False, default=None,
                    help="Path to the directory containing the images")
    ap.add_argument('-o', '--output-dir', type=str, required=False, default=None,
                    help='Path to the output directory where the images will be saved')
    ap.add_argument('-l', '--min-cluster', type=int, required=False, default=2,
                    help='Minimum number of clusters to group the input images')
    ap.add_argument('-t', '--max-cluster', type=int, required=False, default=3,
                    help='Maximum number of cluster to group the input images')
    ap.add_argument('-f', '--model', type=str, required=False, default='resnet50',
                    help='ID of the model to use for feature extraction')
    ap.add_argument('-v', '--version', action='version', version=__version__)
    args = vars(ap.parse_args())

    if args['mode'] == 0:
        logging.info("Running by configuration file")
        run_by_conf_file(args['configuration'])
    elif args['mode'] == 1:
        logging.info("Running by args params")
        cluster_images(args)


if __name__ == '__main__':
    terminal_exec()
