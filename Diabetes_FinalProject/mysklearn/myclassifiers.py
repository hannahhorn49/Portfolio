##############################################
# Programmer: Hannah Horn, Eva Ulrichsen
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #final project
# 12/9/24
# I did not attempt the bonus
# Description: This program contains the various classifer
# implementations
#########################

import operator
from operator import itemgetter
import numpy as np
from mysklearn import utils, myevaluation
import random

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.slope is not None and self.intercept is not None:
            for test_instance in X_test:
                predictions.append(self.slope * test_instance[0] + self.intercept)
        return predictions

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x, y):
        """Fits a simple univariate line y = mx + b to the provided x y data.
        Follows the least squares approach for simple linear regression.

        Args:
            x(list of numeric vals): The list of x values
            y(list of numeric vals): The list of y values

        Returns:
            m(float): The slope of the line fit to x and y
            b(float): The intercept of the line fit to x and y
        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))) \
            / sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        # y = mx + b => y - mx
        b = mean_y - m * mean_x
        return m, b

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
        categorical(list of bool): List of booleans indicating if each feature is categorical (True) or numerical (False)

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3, categorical = None):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
            categorical(list of bool): List of booleans indicating if each feature is categorical (True) or numerical (False)
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        if categorical is not None:
            self.categorical = categorical
        else:
            self.categorical = []

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
                # distance = utils.compute_mixed_euclidean_distance(train_instance, test_instance, self.categorical)
                distance = utils.compute_euclidean_distance(train_instance, test_instance)
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


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        # initialize empty dictionaries to store prior and posterior probabilities
        self.priors = {}
        self.posteriors = {}

        # find total samples in training data
        total_samples = len(y_train)

        # calculate the prior probabilities for each class label
        for label in set(y_train):
            label_count = y_train.count(label)
            self.priors[label] = label_count / total_samples

            # calculate prior probability of the current label
            self.priors[label] = label_count / total_samples

            # initialize a nested dictionary for this label for posteriors
            self.posteriors[label] = {}
        
        # now calculate the posterior probability for each attribute value for each label
        attribute_counts = {}  # this will store the counts of each attribute given class label

        # iterate over both X_train and y_train
        for instance, label in zip(X_train, y_train):
            # if label not present, initialize a new (nested) dictionary
            if label not in attribute_counts:
                attribute_counts[label] = {}
            
            # need to go over each attribute for instance and update counts
            for attribute_index in range(len(instance)):
                attribute_value = instance[attribute_index] # e.g. [1,5] attribute attribute value = 1

            # if not present, initialize a nested dictionary for this feature
                if attribute_index not in attribute_counts[label]:
                    attribute_counts[label][attribute_index] = {}

                # if not present, initialize a nested dictionary for feature value
                if attribute_value not in attribute_counts[label][attribute_index]:
                    attribute_counts[label][attribute_index][attribute_value] = 0
                
                # if does exist, increment count for feature value
                attribute_counts[label][attribute_index][attribute_value] += 1

        # convert these counts to probabilities and store in posteriors
        for label in attribute_counts:
            for attribute_index in attribute_counts[label]:
                total_attribute_count = sum(attribute_counts[label][attribute_index].values())
                self.posteriors[label][attribute_index] = {}

                for attribute_value in attribute_counts[label][attribute_index]:
                    self.posteriors[label][attribute_index][attribute_value] = (
                        attribute_counts[label][attribute_index][attribute_value] / total_attribute_count
                    )
                
                # provide a check and initialize missing expected attribute values to 0 in posteriors
                expected_attribute_values = set()

                for attribute in X_train:
                    expected_attribute_values.add(attribute[attribute_index])
                
                for expected_value in expected_attribute_values:
                    if expected_value not in self.posteriors[label][attribute_index]:
                        self.posteriors[label][attribute_index][expected_value] = 0

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        # need a list to store predictions for each test instance
        y_predicted = []

        # loop through each instance of X_test
        for instance in X_test:
            class_probabilities = {} # stores probability of each class

            # loop through each class label
            for label in self.priors.keys():
                total_probability = self.priors[label]

                # loop through each attribute in the instance with corresponding attribute index
                for attribute_index, attribute_value in enumerate(instance):
                    if attribute_index in self.posteriors[label] and attribute_value in self.posteriors[label][attribute_index]:
                        total_probability *= self.posteriors[label][attribute_index][attribute_value]
                    else:
                        total_probability = 0 # if not found
                        break
                
                # store total probability for current label
                class_probabilities[label] = total_probability
            
            # find class with the highest probability
            predicted_label = max(class_probabilities, key = class_probabilities.get)
            y_predicted.append(predicted_label)
        
        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.used_features = None

    def tdidt(self, current_instances, available_attributes, domains, previous_instances=None, feature_indices=None):
        """Recursively builds a decision tree using the TDIDT algorithm."""

        if previous_instances is None:
            previous_instances = current_instances

        if feature_indices is not None:
            available_attributes = [available_attributes[indice] for indice in feature_indices]
            # print("altered avail att", available_attributes)

        # Get domains for the current instances (possible values for each attribute)
        # domains = self.attribute_domains(current_instances)
        # print("domains: ", domains)
            
        # print("avail attr", available_attributes)

        # Calculate entropy for each attribute and choose the one with the least entropy
        entropies = utils.calculate_entropy(current_instances, available_attributes)
        split_attribute = available_attributes[entropies.index(min(entropies))]

        available_attributes.remove(split_attribute)  # Remove the attribute after use

        # print("Attribute splitting on:", split_attribute)

        # Initialize the tree
        tree = ["Attribute", split_attribute]

        # Partition instances based on the chosen attribute
        partitions = utils.partition_instances(self.header, current_instances, split_attribute, domains)
        # print("partitions:", partitions)
        # print("partition items", sorted(partitions.items()))

        for att_value, att_partition in sorted(partitions.items()): # in sorted order
            value_subtree = ["Value", att_value]
            # print("current partition", att_partition, att_value)

            if len(att_partition) > 0 and utils.all_same_class(att_partition):
                # All instances in this partition have the same class
                label = att_partition[0][-1]
                value_subtree.append(["Leaf", label, len(att_partition), len(current_instances)])

            elif len(att_partition) > 0 and len(available_attributes) == 0:
                # No more attributes to split, resolve with majority vote
                # print("Case 2: No more attributes, resolving clash.")
                majority_label, count = utils.majority_vote(att_partition, previous_instances)
                value_subtree.append(["Leaf", majority_label, len(att_partition), len(current_instances)])

            elif len(att_partition) == 0:
                # Empty partition, resolve with majority vote
                # print("Case 3: Empty partition, resolving with majority vote.")
                majority_label, count = utils.majority_vote(current_instances)
                # print(majority_label)
                tree = ["Leaf", majority_label, len(current_instances), len(previous_instances)]
                break

            else:
                # Continue recursion on the partition with the remaining attributes
                # print(f"Recursing for attribute value {att_value}.")
                subtree = self.tdidt(att_partition, available_attributes.copy(), domains, previous_instances=current_instances)
                value_subtree.append(subtree)

            tree.append(value_subtree)

        return tree

    def fit(self, X_train, y_train, feature_indices=None):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        # print(train)
        # # make a copy a header, b/c python is pass by object reference
        # # and tdidt will be removing attributes from available_attributes
        # header = [f"att{i}" for i in range(len(X_train[0]))]
        # available_attributes = header.copy()
        # tree = tdidt(train, available_attributes)

        # print("tree:", tree)
        self.X_train = X_train
        self.y_train = y_train
        self.header = [f"att{i}" for i in range(len(X_train[0]))]
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        domains = self.attribute_domains(train)
        # print("domains:", domains)
        self.tree = self.tdidt(train, self.header.copy(), domains, feature_indices=feature_indices)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
            Tree (Nested list): tree used for traversal

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for test_instance in X_test:
            # Start at the root of the tree for each test instance
            current_tree = self.tree
            # print("current tree", self.tree)

            while True:
                node_type = current_tree[0]  # "Leaf" or "Attribute"

                if node_type == "Leaf":
                    # Append the predicted class label to y_predicted
                    y_predicted.append(current_tree[1])  # class label
                    break

                # If it's an Attribute node, find the matching branch
                attribute_index = self.header.index(current_tree[1])  # Get the index of the attribute in the test instance
                # print("header", self.header)
                # print("attribute index", attribute_index)
                # print("attr index", attribute_index)
                # print("test instance", test_instance)
                test_value = test_instance[attribute_index]

                # Find the subtree corresponding to the test value
                for subtree in current_tree[2:]:
                    if subtree[1] == test_value:  # Match value
                        current_tree = subtree[2]  # Navigate to the subtree
                        break
                else:
                    # No matching branch found, prediction fails (optional: handle missing values)
                    y_predicted.append(None)  # or handle with a default prediction
                    break

        return y_predicted

    def attribute_domains(self, instances):
        """
        Determines the unique values (domains) for each attribute in the dataset.

        Args:
            instances (list of list): The dataset, where each inner list represents a data instance.
            header (list of str): A list of attribute names, where the index matches the dataset columns.

        Returns:
            dict: A dictionary where the keys are attribute names (from the header)
                and the values are lists of unique values (domains) for each attribute.
        """
        domains = {}
        for col_index, attribute in enumerate(self.header):
            # Extract all unique values in the current column
            unique_values = set(instance[col_index] for instance in instances)
            domains[attribute] = sorted(unique_values)  # Sort for consistency
        return domains

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if not self.tree:
            return

        def traverse_tree(tree, rule_conditions):
            """Recursively traverses the tree to extract decision rules.

            Args:
                tree (list): The current subtree being processed.
                rule_conditions (list): Conditions accumulated along the path to this node.
            """
            node_type = tree[0]

            if node_type == "Leaf":
                # if leaf, print rule
                label = tree[1]
                print("IF " + " AND ".join(rule_conditions) + f" THEN {class_name} = {label}")
                return

            # get the attribute index (tree[1]) and find the corresponding attribute name
            attribute_index = int(tree[1].replace("att", ""))
            attribute_name = (
                attribute_names[attribute_index]
                if attribute_names is not None
                else f"att{attribute_index}"
            )

            # process each subtree for the current attribute
            for subtree in tree[2:]:
                value = subtree[1]
                condition = f"{attribute_name} == {value}"
                traverse_tree(subtree[2], rule_conditions + [condition])

        # traverse from the root of the tree
        traverse_tree(self.tree, [])

class MyRandomForestClassifier:
    def __init__(self, N, M, F, seed_random=None):
        """
            Initializes a RandomForest classifier

            Args:
            - M (int): the number of trees in the forest (best selected)
            - N (int): the number of trees to generate
            - F (?): the subset of attributes to use?
        """

        self.N = N
        self.M = M
        self.F = F
        self.seed_random = seed_random

        self.trees = [] # list to hold the individual trees in the forest
        self.tree_performance = []

    def fit(self, X_train, y_train):
        """
        Args:
        X_train (list): training feature data.
        y_train (list): training labels.
        """
        n_features = len(X_train[0])
        # keep track of previously used feature subsets to prevent duplicates
        used_feature_sets = set()
        # print("y_train", y_train)

        random.seed(self.seed_random)
        
        for i in range(self.N):
            # Bootstrap sampling
            bootstrap_X, out_of_bag_X, bootstrap_y, out_of_bag_y = myevaluation.bootstrap_sample(X_train, y_train, random_state=self.seed_random)

            # Find a unique feature subset
            feature_indices = None
            max_attempts = 100  # Safety limit for retries
            attempts = 0
            while attempts < max_attempts:
                feature_indices = frozenset(random.sample(range(n_features), self.F))  # Use frozenset for immutability
                if feature_indices not in used_feature_sets:
                    used_feature_sets.add(feature_indices)
                    # print("used feature set", used_feature_sets)
                    # create the decision tree classifier and train it on the bootstrap sample and feature subset
                    tree = MyDecisionTreeClassifier()
                    tree.fit(bootstrap_X, bootstrap_y, feature_indices)

                    y_pred = tree.predict(out_of_bag_X)
                    accuracy = myevaluation.accuracy_score(out_of_bag_y, y_pred)

                    self.tree_performance.append((accuracy, tree))
                    break
                attempts += 1

            if attempts == max_attempts:
                raise ValueError("Failed to find a unique feature subset after multiple attempts.")

        # sort the trees by accuracy in descending order
        self.tree_performance.sort(reverse=True, key=itemgetter(0))
        self.trees = []

        # for accuracy, tree in self.tree_performance:
        #     print("Acc and tree:", accuracy, tree.tree)

        for __, tree in self.tree_performance[:self.M]:
            self.trees.append(tree)

    def predict(self, X_test):
        """
        Purpose is to make predictions using the random forest classifier
        Args: X_test (test instances)
        Returns: final predictions (majority vote predictions for each test instance)
        """
        # collect predictions from each tree (calling the decision tree predict method here)
        all_tree_predictions = []

        for test_val in X_test:
            tree_preds = []

            for tree in self.trees: # by the fit method each tree has the format: (tree, feature indices, bootstrap indices)
                # for i in feature_indices:
                #     selected_features.append(X_test[i])
                
                prediction = tree.predict([test_val])[0]
                # print("predictions!!!!!!!!!!!!!!!!!!!!", prediction)
                if prediction is None:
                    prediction = ""
                tree_preds.append(prediction)
            
            # perform majority voting here
            class_counts = {}
            for label in tree_preds:
                if label not in class_counts.keys():
                    class_counts[label] = 0
                class_counts[label] += 1
            
            # determine the majority class
            majority_class = max(class_counts, key=class_counts.get)
            all_tree_predictions.append(majority_class)

        return all_tree_predictions
