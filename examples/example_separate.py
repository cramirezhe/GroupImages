# Created by Carlos Ramirez at 08/10/2022
import numpy as np

from group_images.separate import Separate

# Create a dummy directory of false image features
print("[INFO] Creating fake images and features")
images_feat = {}
for idx in range(1000):
    images_feat[f"{idx}_ex.png"] = np.random.rand(1024)

print("[INFO] Clustering images")
selector = Separate(images_feat, min_cluster=2, max_cluster=100)
result = selector.cluster_images()

print("[INFO] Printing result:")
for key, value in result.items():
    print(f"Image {key} goes to cluster {value}")
