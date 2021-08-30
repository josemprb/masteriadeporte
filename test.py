import unittest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.datasets import make_blobs
import numpy as np

from knn import Knn


iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=22)

"""
X, y = make_blobs(n_samples=200, centers=2, n_features=20, random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
"""
little_X = np.array([[2, 2], [9, 9]])
little_Y = np.array([1, 1])


class KnnTest(unittest.TestCase):
    def test_euclidean_distance(self):
        """Test to check that euclidean distance is correct"""
        knn = Knn(n_neighbors=3)
        knn.fit(np.array(little_X), little_Y)
        d = knn._euclidean_distance(np.array([5, 6]))
        assert (d == [5, 5]).all(), "Euclidean Distance is not correct"

    def test_manhattan_distance(self):
        """Test to check that manhattan distance is correct"""
        knn = Knn(n_neighbors=3)
        knn.fit(np.array(little_X), little_Y)
        d = knn._manhattan_distance(np.array([5, 6]))
        assert (d == [7, 7]).all(), "Manhattan Distance is not correct"

    def test_minkowski_distance(self):
        """Test to check that minkowski distance is correct"""
        knn = Knn(n_neighbors=3, p=5)
        knn.fit(np.array(little_X), little_Y)
        d = knn._minkowski_distance(np.array([3, 4]))
        assert np.allclose(d, [2.01234, 6.419382]), "Minkowski Distance is not correct"

    def test_input_dimension(self):
        """Test to check that we raise an exception id X and y dimension are nor consistent"""
        knn = Knn(n_neighbors=3)
        with self.assertRaises(ValueError): knn.fit(X_train, y_test)

    def test_uniform_weight(self):
        """Test to see that _uniform_weights return a weight of 1 for each distance"""
        knn = Knn(n_neighbors=3)
        distances = np.array([2, .3, 4])
        weights = knn._uniform_weights(distances)
        assert np.allclose(weights, np.array([[1, 2], [1, .3], [1, 4]])), "uniform_weights are not correct"

    def test_distance_weight(self):
        """Test to see that _distance_weights return a weight of 1/d for each distance"""
        knn = Knn(n_neighbors=3)
        distances = np.array([2, .3, 4])
        weights = knn._distance_weights(distances)
        assert np.allclose(weights, np.array([[1/2, 2], [1/.3, .3], [1/4, 4]])), "distance_weights are not correct"

    def test_uniform_weight_with_0(self):
        """Test to see that _distance_weights return a weight of 1/d for each distance"""
        knn = Knn(n_neighbors=3)
        distances = np.array([0, .3, 4])
        weights = knn._distance_weights(distances)
        assert np.allclose(weights, np.array([[1, 0], [1/.3, .3], [1/4, 4]])), "distance_weights are not correct when we have distances of 0"

    def test_k_1(self):
        """Test to compare our knn with Sklearn knn when k=1"""
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)

        knn2 = Knn(n_neighbors=1)
        knn2.fit(X_train, y_train)
        prediction2 = knn2.predict(X_test)

        assert np.alltrue(prediction == prediction2), "Error testing knn with k=1"

    def test_k_5(self):
        """Test to compare our knn with Sklearn knn when k=5 and distance is euclidean"""
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)

        knn2 = Knn(n_neighbors=5)
        knn2.fit(X_train, y_train)
        prediction2 = knn2.predict(X_test)

        assert np.alltrue(prediction == prediction2), "Error testing knn with k=5"

    def test_k_5_distance_manhattan(self):
        """Test to compare our knn with Sklearn knn when k=5 and distance is manhattan"""
        knn = KNeighborsClassifier(n_neighbors=5, metric="manhattan")
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)

        knn2 = Knn(n_neighbors=5, metric="manhattan")
        knn2.fit(X_train, y_train)
        prediction2 = knn2.predict(X_test)

        assert np.alltrue(prediction == prediction2), "Error testing knn (manhattan) with k=5"

    def test_k_5_distance_minkowski(self):
        """Test to compare our knn with Sklearn knn when k=5 and distance is minkowski with p=3"""
        knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=3)
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)

        knn2 = Knn(n_neighbors=5, metric="minkowski", p=3)
        knn2.fit(X_train, y_train)
        prediction2 = knn2.predict(X_test)

        assert np.alltrue(prediction == prediction2), "Error testing knn (minkowski) with k=5 and p=3"

    def test_non_valid_metric(self):
        """Test to check that we raise an exception id metric is not in our use case"""
        with self.assertRaises(ValueError): Knn(n_neighbors=5, metric="chebyshev")

    def test_distance_weight_2(self):
        """Test to compare our knn with Sklearn when k=15 and weights are the inverse of distance"""
        knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
        knn.fit(X_train, y_train)
        prediction = knn.predict(X_test)

        knn2 = Knn(n_neighbors=15, weights='distance')
        knn2.fit(X_train, y_train)
        prediction2 = knn2.predict(X_test)

        assert np.alltrue(prediction == prediction2), "Error testing knn with k=5 and weights=distance"


if __name__ == '__main__':
    unittest.main()
