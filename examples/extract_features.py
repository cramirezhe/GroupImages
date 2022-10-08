# Created by Carlos Ramirez at 08/10/2022
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from group_images.feature_extractor import FeatureExtractor

feat_ext = FeatureExtractor("../tmp/dir", "efficientnetv2_b1", pooling='max')

images_feat = feat_ext.get_features(batch_size=8)

# Get the features from our dictionary as a list
features = list(images_feat.values())

# Cast it to numpy array
features = np.array(features)

kmeans = MiniBatchKMeans(n_clusters=3)

kmeans.fit(features)

out_dict = {}
for idx, (image_path, feature) in enumerate(images_feat.items()):
    out_dict[image_path] = kmeans.labels_[idx]

