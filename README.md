[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">


  <h3 align="center">Group Images</h3>

  <p align="center">
    Cluster unsorted images using Deep Learning and KMeans
    <br />
   </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About the project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#pip-installation">pip installation</a></li>
        <li><a href="#from-source-code">From source code</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="##terminal-mode">Terminal mode</a></li>
        <li><a href="##package">Package</a></li>
      </ul>
    </li>
  </ol>
</details>


## About the project

![Product Name Screen Shot][product-screenshot]

This project aims to cluster unsorted images in a given directory, the images can be mixed
or inside subdirectories.

This package will look for the images and cluster them, so you can organize your photo library or
pre-sort a dataset into categories.

I have found multiple examples of doing this in jupyter notebooks and internet blogs,
and I would like to create a repo that can be reused to perform this without the need of
going into the technical details or implementing these operations from scratch every time you
want to cluster your images.

In addition, this project will look for the optimal number of cluster for your use case,
you will need to provide a minimum and maximum number of clusters.

I include 2 modes of use:

1. Terminal mode: in this mode you can run the clustering without writing a single line of code
2. Using functions: in this mode you can explore the different available functions and tune the parameters
at your convenience.

I really hope you enjoy this little contribution as I did enjoy while coding it :smile:.

## Getting Started

Before continuing you must be sure that you have installed pip and that you are using Python 3.8.

You can install this project from the source or using pip.

### pip installation
```shell
$ pip install GroupImages
```

### From source code
1. Clone the repository
```shell
$ git clone https://github.com/cramirezhe/GroupImages
```
2. Install requirements
```shell
$ pip install -r requirements.txt
```
4. (Optional) Test the package
```shell
$ tox
```
5. Install the package
```shell
$ python setup.py install
```

## Usage

### Terminal mode

This mode allows you to run the clustering without writing any code.

To run it you just need to open a terminal and tap "cluster_images".

```sh
$ cluster_images --help
usage: cluster_images [-h] -m MODE [-c CONFIGURATION] [-i INPUT_DIR] [-o OUTPUT_DIR] [-l MIN_CLUSTER] [-t MAX_CLUSTER] [-f MODEL] [-v]

Cluster images in a given directory

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Select mode to use: 0= by configuration file, 1= by args parameters
  -c CONFIGURATION, --configuration CONFIGURATION
                        Path to the configuration file
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Path to the directory containing the images
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to the output directory where the images will be saved
  -l MIN_CLUSTER, --min-cluster MIN_CLUSTER
                        Minimum number of clusters to group the input images
  -t MAX_CLUSTER, --max-cluster MAX_CLUSTER
                        Maximum number of cluster to group the input images
  -f MODEL, --model MODEL
                        ID of the model to use for feature extraction
  -v, --version         show program's version number and exit
```

In the most basic mode, you just need to provide:
* mode: set it to one
* input-dir: this is the path to the directory containing images
* output-dir: where the cluster images will be stored
* min-cluster: minimum number of clusters (must be > 2)
* max-cluster: maximum number of clusters (must be > min-cluster)

```sh
$ cluster_images --mode 1 --input-dir GroupImages/test_imgs --output-dir GroupImages/output --min-cluster 2 --max-cluster 4
Metal device set to: Apple M1

systemMemory: 8.00 GB
maxCacheSize: 2.67 GB

2022-09-21 19:45:36.863841: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2022-09-21 19:45:36.864412: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2022-09-21 19:45:37.821342: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
2022-09-21 19:45:38.312615: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:113] Plugin optimizer for device_type GPU is enabled.
1/1 [==============================] - 1s 628ms/step
1/1 [==============================] - 0s 25ms/step
1/1 [==============================] - 0s 201ms/step
```

Finally, you can explore the results:

```sh
$ tree GroupImages/output
GroupImages/output
├── 0
│   ├── 0000.jpeg
│   ├── 0001.jpeg
│   ├── 0005.jpeg
│   ├── 0006.jpeg
│   └── 0007.jpeg
├── 1
│   ├── 0013.jpeg
│   ├── 0014.jpeg
│   └── 0015.jpeg
├── 2
│   ├── 0002.jpeg
│   ├── 0009.jpeg
│   ├── 0010.jpeg
│   ├── 0011.jpeg
│   ├── 0012.jpeg
│   └── 0016.jpeg
└── 3
    ├── 0003.jpeg
    ├── 0004.jpeg
    └── 0008.jpeg
```

For more experienced users you can select the feature extractor:

* model: Deep Learning model used for feature extraction, some options are:
  (resnet50, resnet101, resnet152, densenet121, mobilenet_v2, inceptionv3)

If you want to explore other parameters you can use a configuration file instead
of argument parameters:

```sh
# You can find an example inside tests/configuration.ini
$ cluster_images --mode 1 --configuration PATH/TO/conf.ini
```

Example:

```ini
[GENERAL]
INPUT_PATH=PATH2IMAGES
OUTPUT_PATH=RESULT_DIR

[FEAT_EXT]
; Models: 
; ['densenet121', 'densenet169', 'densenet201', 'efficientnetv2_b0', 'efficientnetv2_b1',
;  'efficientnetv2_b2', 'inceptionv3', 'mobilenet_v2', 'resnet50', 'resnet101', 'resnet152']
MODEL=resnet50 ; Deep learning model to use for Feature Extraction
POOLING=avg ; Pooling for last layer of the feature extractor [avg, max, None]
BATCH_SIZE=8 ; Batch size to use for inference (affected by the amount of free memory)

[CLUSTER]
MIN_CLUSTER=2 ; Minimum number of clusters
MAX_CLUSTER=4 ; Maximum number of clusters to search KMeans
ITERATIONS=100 ; Number of iteration for KMeans algorithm
RANDOM_STATE=42
STOP_INERTIA=5000 ; Stop searching for optimal cluster when inertia is below this value
```

### Package

For more control on the application, you can directly access to the classes and functions by
importing "****group_images****":

```python
import group_images
```

For example, you can use it to extract features from all the images in a directory and use
other algorithm to cluster them:

```python
from group_images.feature_extractor import FeatureExtractor

fe = FeatureExtractor('./mes_images', 'inceptionv3', pooling='max')
# Image is a dictionary where the key is the image path and the content is the feature vector
images_feat = fe.get_features()
```

Or if you prefer, you can use your own feature extractor to cluster the images

```python
from group_images.separate import Separate

def group_images(feature_images, min_cluster, max_cluster):
    s = Separate(feature_images, min_cluster, max_cluster)
    # Return a dictionary with image path as key and the cluster where it was sorted.
    # Optionally you can visualize the cluster by using the experimental feature:
    s.plot_vectors()
    return s.cluster_images()
```

For more information and tips, please refer to the documentation in the 'docs' folder or the
examples in the examples' directory.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/cramirezhe/GroupImages.svg?style=for-the-badge
[contributors-url]: https://github.com/cramirezhe/GroupImages/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/cramirezhe/GroupImages.svg?style=for-the-badge
[forks-url]: https://github.com/cramirezhe/GroupImages/network/members
[stars-shield]: https://img.shields.io/github/stars/cramirezhe/GroupImages.svg?style=for-the-badge
[stars-url]: https://github.com/cramirezhe/GroupImages/stargazers
[issues-shield]: https://img.shields.io/github/issues/cramirezhe/GroupImages.svg?style=for-the-badge
[issues-url]: https://github.com/cramirezhe/GroupImages/issues
[license-shield]: https://img.shields.io/github/license/cramirezhe/GroupImages.svg?style=for-the-badge
[license-url]: https://github.com/cramirezhe/GroupImages/blob/main/LICENSE
[product-screenshot]: images/Plot_Example.png
