import numpy as np


class Knn:
    def __init__(self, n_neighbors, metric='minkowski', p=2, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.p = p
        self.metric = metric
        self.weights = weights
        if metric not in ['minkowski', 'euclidean', 'manhattan']:
            raise ValueError
        if weights not in ['uniform', 'distance']:
            raise ValueError

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError
        self.X = X
        self.y = y

    def _manhattan_distance(self, point):
        return np.sum(abs(self.X - point), axis=1)

    def _euclidean_distance(self, point):
        return np.sum((self.X - point)**2, axis=1)**0.5

    def _minkowski_distance(self, point):
        return np.sum(abs(self.X - point)**self.p, axis=1)**(1/self.p)

    def _uniform_weights(self, distances):
        return np.array([(1, y) for d, y in np.ndenumerate(distances)])

    def _distance_weights(self, distances):
        return np.array([(1/y if y else 1, y) for d, y in np.ndenumerate(distances)])

    def _predict_point(self, point):
        if self.metric == 'minkowski':
            distances = self._minkowski_distance(point)
        elif self.metric == 'euclidean':
            distances = self._euclidean_distance(point)
        else:
            distances = self._manhattan_distance(point)
        indices = np.argsort(distances, axis=0)
        labels = np.unique(self.y[indices[:self.n_neighbors]])
        if len(labels) == 1:
            label = np.unique(labels)[0]
        else:
            counts = np.zeros(max(labels) + 1)
            for i in range(self.n_neighbors):
                if self.weights == 'uniform':
                    weight = self._uniform_weights(distances[indices[i]])
                else:
                    weight = self._distance_weights(distances[indices[i]])
                counts[self.y[indices[i]]] += weight[0][0]
            label = np.argmax(counts)
        return label

    def predict(self, x):
        labels = []
        for point in x:
            labels.append(self._predict_point(point))
        return labels
