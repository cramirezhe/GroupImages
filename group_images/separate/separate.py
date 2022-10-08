# Created by Carlos Ramirez at 12/09/2022
import logging
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class Separate:
    def __init__(self, dict_imgs: dict, min_cluster: int = 2, max_cluster: int = 3):
        """
        Initializes the class that will separate images in clusters.
        Args:
            dict_imgs (dict): dictionary in format key=image_path value=image_features
            min_cluster (int): start point to search for optimal # of cluster
            max_cluster (int): limit # of clusters
        """
        self._dict_imgs = dict_imgs
        # Verify min and max values
        self._min_cluster = max(2, min_cluster)
        self._max_cluster = max_cluster
        self._best_model = None  # type: Optional[KMeans]
        if self._min_cluster > self._max_cluster:
            tmp = self._min_cluster
            self._min_cluster = self._max_cluster
            self._max_cluster = tmp
        keys = list(self._dict_imgs.keys())
        # Max cluster must be smaller than the number of images
        if len(keys) <= self._max_cluster:
            raise ValueError("max_cluster must be smaller than the number of images")

    def update_images(self, dict_imgs: dict) -> None:
        """Updates the dictionary of images"""
        if not isinstance(dict_imgs, dict):
            raise ValueError("Please provide a dict with the format "
                             "key=image_path, value=image features")
        self._dict_imgs = dict_imgs

    def set_min_cluster(self, min_cluster: int) -> None:
        """Updates the start point to look for clusters, raises ValueError in case of error"""
        if min_cluster <= 0:
            raise ValueError("Please use a positive number to update min_cluster")
        if min_cluster >= self._max_cluster:
            logging.warning(f"{min_cluster} is greater than current limit, update first"
                            f"max_cluster before continuing")
        else:
            self._min_cluster = min_cluster

    def set_max_cluster(self, max_cluster: int) -> None:
        """Update limit # of clusters, raises ValueError in case of error"""
        if max_cluster <= 0:
            raise ValueError("Please use a positive number to update max_cluster")
        if max_cluster <= self._min_cluster:
            logging.warning(f"{max_cluster} is smaller than min_cluster, update first"
                            f"min_cluster before continuing")
        else:
            self._max_cluster = max_cluster

    def get_max_cluster(self) -> int:
        """Get current limit of clusters"""
        return self._max_cluster

    def get_min_cluster(self) -> int:
        """Get current minimum # of clusters"""
        return self._min_cluster

    def get_dict_imgs(self) -> dict:
        """Return dictionary of images -> feature vectors"""
        return self._dict_imgs

    def cluster_images(self, iterations: int = 100, random_state: Optional[int] = None,
                       early_stop_inertia: float = 0.0) -> dict:
        """
        Search for the optimal # of clusters for a given set of images
        Args:
            iterations (int): # of iteration to run clustering fit algorithm
            random_state (Optional[int]): random_state to reproduce result, by default it is None
            early_stop_inertia (float): value to stop looking for optimal cluster if cluster
                                        inertia is smaller than this value

        Returns:
            dict: A dictionary with key=image_path and value=matching cluster
        """
        # Get the features from our dictionary as a list
        features = list(self._dict_imgs.values())
        # Cast it to numpy array
        features = np.array(features)
        best_inertia = np.inf
        best_model = None
        best_cluster = 0
        for n_clusters in range(self._min_cluster, self._max_cluster + 1):
            logging.info(f"Trying {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, max_iter=iterations,
                            random_state=random_state)
            kmeans.fit(features)
            logging.info(f"\t Inertia for cluster was: {kmeans.inertia_}")
            if best_inertia > kmeans.inertia_ or early_stop_inertia > kmeans.inertia_:
                # Update to best cluster
                best_model = kmeans
                best_inertia = kmeans.inertia_
                best_cluster = n_clusters
        # Update our best model
        self._best_model = best_model
        logging.info(f"Best cluster is {best_cluster} with inertia {best_inertia}")
        out_dict = {}
        for idx, (image_path, feature) in enumerate(self._dict_imgs.items()):
            out_dict[image_path] = self._best_model.labels_[idx]
        return out_dict

    def _get_examples_scatter(self):
        """Randomly select one element from our database to draw in the scatterplot per cluster"""
        labels = self._best_model.labels_
        # Grab all examples
        all_examples = {}
        for idx, (image_path, _) in enumerate(self._dict_imgs.items()):
            current_label = labels[idx]
            if all_examples.get(current_label, None) is None:
                all_examples[current_label] = [image_path]
            else:
                all_examples[current_label].append(image_path)
        # Select randomly one element
        examples = {}
        for label, images_paths in all_examples.items():
            examples[label] = {
                'image_path': random.choice(images_paths),
                'max_x': -np.inf,
                'min_x': np.inf,
                'max_y': -np.inf,
                'min_y': np.inf
            }
        return examples

    def _draw_examples_plot(self, ax, z, zoom):
        """Draw our selected example from _get_examples_scatter in our plot"""
        labels = self._best_model.labels_
        examples = self._get_examples_scatter()
        for idx in range(len(labels)):
            examples[labels[idx]]['max_x'] = max(examples[labels[idx]]['max_x'], z[idx, 0])
            examples[labels[idx]]['min_x'] = min(examples[labels[idx]]['min_x'], z[idx, 0])
            examples[labels[idx]]['max_y'] = max(examples[labels[idx]]['max_y'], z[idx, 1])
            examples[labels[idx]]['min_y'] = min(examples[labels[idx]]['min_x'], z[idx, 1])
        for label, metadata in examples.items():
            x_coor = metadata['min_x'] + (metadata['max_x'] - metadata['min_x']) / 2
            y_coor = metadata['min_y'] + (metadata['max_y'] - metadata['min_y']) / 2
            image = plt.imread(metadata['image_path'])
            im = OffsetImage(image, zoom=zoom)
            ab = AnnotationBbox(im, (x_coor, y_coor), frameon=False)
            ax.add_artist(ab)

    def plot_vectors(self, random_state: Optional[int] = None,
                     zoom: Optional[float] = 0.05) -> None:
        """
        Experimental feature, plot images features in a 2d graph with an image in the
        Args:
            random_state Optional[int]: optional int value to get random values
            zoom: zoom to use in order to draw images in the plot values smaller than 1.0
                  will decrease the images, otherwise it will increase

        Returns:
            None
        """
        if self._best_model is None:
            logging.warning("Please fit the data before plotting it.")
            return
        if zoom <= 0.0:
            raise ValueError("zoom parameter must be positive")
        labels = self._best_model.labels_
        n_clusters = self._best_model.cluster_centers_.shape[0]
        # Get the features from our dictionary as a list
        features = list(self._dict_imgs.values())
        # Cast it to numpy array
        features = np.array(features)
        tsne = TSNE(n_components=2, verbose=0, random_state=random_state,
                    perplexity=features.shape[0] / n_clusters)
        # fit the data
        z = tsne.fit_transform(features)
        # if everything work correctly, grab an example of every cluster
        fig, ax = plt.subplots()
        ax.scatter(x=z[:, 0], y=z[:, 1], c=labels)
        ax.set_title('Clusters in data')
        ax.set(xlabel='comp-1', ylabel='comp-2')
        # Draw images in plot
        self._draw_examples_plot(ax, z, zoom)
        plt.show()
