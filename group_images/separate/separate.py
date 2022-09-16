# Created by Carlos Ramirez at 12/09/2022
import logging
from typing import Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


class Separate:
    def __init__(self, dict_imgs: dict, min_cluster: int = 2, max_cluster: int = 3):
        """
        Initializes the class that will separate images in clusters.
        :param dict_imgs: dictionary in format key=image_path value=image_features
        :param min_cluster: start point to search for optimal # of cluster
        :param max_cluster: limit # of clusters
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
            raise ValueError(f"max_cluster must be smaller than the number of images")

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
        :param iterations: # of iteration to run clustering fit algorithm
        :param random_state: random_state to reproduce result, by default it is None
        :param early_stop_inertia: value to stop looking for optimal cluster if cluster inertia
                                   is smaller than this value
        :return:
            A dictionary with key=image_path and value=matching cluster
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
            out_dict[image_path] = best_model.labels_[idx]
        return out_dict

    def plot_vectors(self, random_state=None):
        """Experimental, plot images features in a 2d graph"""
        if self._best_model is None:
            logging.warning("Please fit the data before plotting it.")
            return
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
        df = pd.DataFrame()
        df["y"] = labels
        print(type(z), z.shape)
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls",
                                                  n_clusters),
                        data=df).set(title="Clustered images")
        plt.show()

