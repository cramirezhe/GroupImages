# Created by Carlos Ramirez at 20/09/2022
import logging
import random
import unittest
import numpy as np

from group_images.separate import Separate


class TestSeparator(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_dict = self._generate_random_dict()

    def _generate_random_dict(self, n_samples: int = 100) -> dict:
        """Generate a fake dictionary images -> features"""
        base_name = 'img'
        test_dict = {}
        for idx in range(n_samples):
            test_dict[f"{base_name}_{str(idx).zfill(4)}.png"] = np.random.randn(1, 2048)
        return test_dict

    def test_update_images(self):
        """Test update dictionary images -> features"""
        _cluster = Separate(self._test_dict)
        new_dict = self._generate_random_dict(50)
        _cluster.update_images(new_dict)
        cluster_dict = _cluster.get_dict_imgs()
        # First verify if dict was updated
        self.assertEqual(len(new_dict), len(cluster_dict))
        for key, value in new_dict.items():
            self.assertIsNotNone(cluster_dict.get(key, None))
            result = np.sum(cluster_dict[key] - value)
            self.assertEqual(result, 0)

    def test_update_clusters(self):
        """Test a valid update of clusters sizes"""
        initial_min, initial_max = 2, 3
        new_min = 100
        new_max = 0
        while new_min > new_max or new_max < initial_max or new_min < initial_min:
            new_min = random.randint(2, len(list(self._test_dict.keys())) - 2)
            new_max = random.randint(2, len(list(self._test_dict.keys())) - 2)
        _cluster = Separate(self._test_dict, initial_min, initial_max)
        _cluster.set_max_cluster(new_max)
        self.assertEqual(new_max, _cluster.get_max_cluster())
        _cluster.set_min_cluster(new_min)
        self.assertEqual(new_min, _cluster.get_min_cluster())

    def test_invalid_clusters(self):
        """Test an invalid update of clusters sizes"""
        initial_min, initial_max = 20, 30
        new_min = 100
        new_max = 10
        _cluster = Separate(self._test_dict, initial_min, initial_max)
        _cluster.set_max_cluster(new_max)
        self.assertNotEqual(new_max, _cluster.get_max_cluster())
        _cluster.set_min_cluster(new_min)
        self.assertNotEqual(new_min, _cluster.get_min_cluster())


if __name__ == '__main__':
    unittest.main()
