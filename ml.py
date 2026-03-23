import math
import numpy as np
from collections import Counter


def calc_mean(values):
    return sum(values) / len(values)


def calc_standard_deviation(values, mean):
    sum_diff = 0
    for v in values:
        sum_diff += math.pow(v - mean, 2)

    return math.sqrt(sum_diff / (len(values) - 1))


def normalize_data(X):
    normalized_data = np.zeros(X.shape)
    for row_i in range(len(X)):
        row = X[row_i]
        mean = calc_mean(row)
        std = calc_standard_deviation(row, mean)

        normalized_col = np.zeros(row.shape)
        for i in range(len(row)):
            normalized_col[i] = (row[i] - mean) / std

        normalized_data[row_i] = normalized_col
    return normalized_data


def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance


def extract_data_from_file(fileName, y_index):
    file = open(fileName)
    data = list(map(lambda line: list(map(int, filter(lambda x: x != '', line.split(' ')))), file.readlines()))
    X = np.asarray(list(map(lambda x: np.asarray(x[:y_index]), data)))
    y = np.asarray(list(map(lambda x: np.asarray(x[y_index]), data)))
    file.close()

    return normalize_data(X[:500]), y[:500]


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, y):
        correct_predictions_total = 0
        for row_i in range(len(X)):
            row = X[row_i]
            predicted_class = self._predict(row)
            actual_class = y[row_i]
            is_correct_prediction = predicted_class == actual_class
            if is_correct_prediction:
                correct_predictions_total += 1
            print(
                f'K: {self.k} | Row: #{row_i+1} | Predicted: {predicted_class} | Actual: {actual_class} {"✓" if is_correct_prediction else "✗"}')
        accuracy = round(correct_predictions_total / len(y) * 100, 2)
        return accuracy

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]


classes_index = 16
X_train, y_train = extract_data_from_file('pendigits_training.txt', classes_index)
X_test, y_test = extract_data_from_file('pendigits_test.txt', classes_index)

result = {}
for k in range(1, 10):
    knn = KNN(k)
    knn.fit(X_train, y_train)
    acc = knn.predict(X_test, y_test)
    result[k] = acc

print('Done.\n')

for k in result.keys():
    acc = result[k]
    print(f'K: {k} | Correct: {acc} | All: {len(y_test)} | Acc: {acc}')
