import logging
import unittest

from group_images.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._imgs_dir = 'test_imgs'
        self._models = list(FeatureExtractor.models.keys())
        logging.getLogger().setLevel(logging.INFO)

    def test_find_images(self):
        """Test finding images in a directory"""
        feat_ext = FeatureExtractor(self._imgs_dir)
        images_paths = feat_ext.find_images_dir()
        self.assertEqual(len(images_paths), 17)

    def test_feature_extractor(self):
        """Test feature extraction using different pooling techniques"""
        poolings = ['avg', 'max', None]
        for model_id in self._models:
            for pool in poolings:
                logging.info(f"Testing feature extraction for {model_id} with pooling {pool}")
                feat_ext = FeatureExtractor(self._imgs_dir, model_id, pool)
                images_paths = feat_ext.find_images_dir()
                batch_size = len(images_paths) // 2
                result = feat_ext.get_features(batch_size)
                for image_path, feats in result.items():
                    self.assertIsNotNone(feats)
                    self.assertGreaterEqual(feats.shape[0], 1, "Feature vector is empty")


if __name__ == '__main__':
    unittest.main()
