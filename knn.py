# KNN Classifier from scratch with pandas dataframe
from collections import defaultdict
import pandas as pd


class KNN:

    def __init__(self, k=3, distance="euclidean", p=3, weighted=False):
        self.k = k
        self.distance = distance
        self.p = p
        self.weighted = weighted

    def fit(self, X, y):
        self.X_train = X.to_numpy() if hasattr(X, "to_numpy") else X
        self.y_train = y.to_numpy() if hasattr(y, "to_numpy") else y

    def predict(self, X):
        X = X.to_numpy() if hasattr(X, "to_numpy") else X
        return [self._predict(x) for x in X]

    def _predict(self, x):
        # Convert x to numpy array if it's a Series
        x = x.values if hasattr(x, "values") else x

        # Compute distances
        if self.distance == "euclidean":
            # Calculate distances from x to all training points
            distances = [
                self._euclidean_distance(
                    x, x_train.values if hasattr(x_train, "values") else x_train
                )
                for x_train in self.X_train
            ]

        elif self.distance == "manhattan":
            distances = [
                self._manhattan_distance(
                    x, x_train.values if hasattr(x_train, "values") else x_train
                )
                for x_train in self.X_train
            ]

        elif self.distance == "minkowski":
            distances = [
                self._minkowski_distance(
                    x, x_train.values if hasattr(x_train, "values") else x_train, self.p
                )
                for x_train in self.X_train
            ]

        else:
            raise ValueError(
                "Distance metric not recognized. Use 'euclidean', 'manhattan', or 'minkowski'."
            )

        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]

        if not self.weighted:
            # Majority vote
            return max(set(k_nearest_labels), key=k_nearest_labels.count)
        else:
            # Weighted vote
            weights = defaultdict(float)
            for label, dist in zip(k_nearest_labels, k_nearest_distances):
                weight = 1 / (dist + 1e-5)  # avoid division by zero
                weights[label] += weight
            return max(weights, key=weights.get)

    def score(self, X, y):
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)

    def get_params(self):
        return {"k": self.k, "distance": self.distance, "p": self.p}

    @staticmethod
    def _euclidean_distance(x1, x2):
        return sum((a - b) ** 2 for a, b in zip(x1, x2)) ** 0.5

    @staticmethod
    def _manhattan_distance(x1, x2):
        return sum(abs(a - b) for a, b in zip(x1, x2))

    @staticmethod
    def _minkowski_distance(x1, x2, p=3):
        return sum(abs(a - b) ** p for a, b in zip(x1, x2)) ** (1 / p)
