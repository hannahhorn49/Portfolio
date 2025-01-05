##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01, Fall 2024
# Programming Assignment #5
# 10/28/24
# I did not attempt the bonus
#
# Description: This program provides the implementation that to pass the tests
#               in test_myclassifiers for using these classifiers for prediction.
##############################################
"""Module providing access to the MySimpleLinearRegressor class as well as other reusable functions"""
import operator
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn import myutils

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, x_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """

        # first check if there is an instance of MySimpleLinearRegressor
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        # directly fit the regressor using the training data
        self.regressor.fit(x_train, y_train)

    def predict(self, x_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.regressor is None:
            raise ValueError("Regressor is not initalized. Need to call fit() before predict().")
        predictions = []
        numeric_predictions = self.regressor.predict(x_test)
        for prediction in numeric_predictions:
            discretized_value = self.discretizer(prediction)
            predictions.append(discretized_value)
        return predictions

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = x_train
        self.y_train = y_train

    def kneighbors(self, x_test, n_neighbors=None):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        if n_neighbors is None:
            n_neighbors = self.n_neighbors # use default (3) if not specified

        # need to store distances and index for for each test instance (return two separate lists)
        indexes = []
        distances = []

        for test_instance in x_test:
            row_index_distance = [] # list to store distances for current test instance

            for i, train_instance in enumerate(self.X_train): # calculate distance between test instance and all training instances
                distance = myutils.compute_euclidean_distance(train_instance, test_instance)
                row_index_distance.append((i, distance))
            # sort distances and get the k closest neighbors
            row_index_distance.sort(key = operator.itemgetter(-1))
            top_k = row_index_distance[:n_neighbors] # use n_neighbors from parameter

            current_distances = []
            current_indexes = []

            for value in top_k:
                current_indexes.append(value[0])
                current_distances.append(value[1])
            indexes.append(current_indexes)
            distances.append(current_distances)
        return indexes, distances

    def predict(self, x_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # need to call the kneighbors method to get the index of the k nearest neighbors
        neighbor_indexes, distances = self.kneighbors(x_test)
        y_predicted = []

        # next need to find the corresponding y_train values for the nearest neighbors (classification)
        # perform majority voting to determine most common class label among neighbors
        # return the predicted class label

        for i, neighbors in enumerate(neighbor_indexes):
            label_counts = {}
            for neighbor_index in neighbors:
                # look up the class label in y_train for the current neighbor
                label = self.y_train[neighbor_index]

                # count occurence of each label
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            # do majority vote
            most_common_label = max(label_counts, key = label_counts.get)
            y_predicted.append(most_common_label)
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        # should use a dictionary (class:count) so can count occurences of each instance of a class
        class_count = {}

        # should count occurences of each class label in y_train
        for class_label in y_train:
            if class_label in class_count:
                class_count[class_label] += 1
            else:
                class_count[class_label] = 1
        # find class with most occurences
        self.most_common_label = max(class_count, key = class_count.get)

    def predict(self, x_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        # should use whatever is stored in the most common label to predict all instances
        predictions = [self.most_common_label] * len(x_test)
        return predictions
