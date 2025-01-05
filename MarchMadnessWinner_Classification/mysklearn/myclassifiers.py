##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #7
# 11/20/24
# I did not attempt the bonus
# Description: This program contains the various classifer
# implementations
##############################################

import operator
import math
import numpy as np
from mysklearn import myutils

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
        self.categorical = categorical

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
                distance = myutils.compute_mixed_euclidean_distance(train_instance, test_instance, self.categorical)
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
        self.header = []
        self.attribute_domains = {}
    
    def partition_instances(self, instances, attribute):
        """This function places instances into a partition 
    
        Args:
            instances
            attribute
        
        Returns: 
            parition containing instances
        """
        # this is group by attribute domain (not values of attribute in instances)
        # lets use dictionaries
        att_index = self.header.index(attribute)
        att_domain = self.attribute_domains[attribute]
        partitions = {}
        for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions
    
    def calculate_entropy(self, class_labels):
        """Calculate the entropy of a set of class labels."""
        class_counts = {}
        for label in class_labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        total_instances = len(class_labels)
        entropy = 0
        for count in class_counts.values():
            prob = count / total_instances
            entropy -= prob * math.log2(prob)  # Entropy formula
        
        return entropy
    
    def calculate_entropy_of_partition(self, instances, attribute):
        """
        Calculate the weighted entropy of the dataset partitioned by a specific attribute.
        
        Args:
        - instances (list): The dataset of instances.
        - attribute (str): The attribute to split the data on.
        - header (list): The list of attribute names corresponding to the columns of the instances.
        
        Returns:
        - weighted_entropy (float): The weighted average entropy for the partitions created by the attribute.
        """
        # Step 1: Partition the instances based on the selected attribute
        att_partitions = self.partition_instances(instances, attribute)
        
        # Step 2: Calculate the weighted entropy
        total_instances = len(instances)
        weighted_entropy = 0
        
        # For each partition (e.g., "Junior", "Mid", "Senior")
        for att_value, att_partition in att_partitions.items():
            # Get the class labels for the current partition
            class_labels = [instance[-1] for instance in att_partition]
            
            # Calculate the entropy for this partition
            partition_entropy = self.calculate_entropy(class_labels)
            
            # Weight for this partition (proportion of total instances)
            partition_weight = len(att_partition) / total_instances
            
            # Add the weighted entropy of the current partition to the total
            weighted_entropy += partition_weight * partition_entropy
        
        return weighted_entropy
    
    def select_attribute(self, instances, available_attributes):
    
        """Select the attribute with the smallest weighted entropy (Enew)."""
        min_entropy = float('inf')  # as a placeholder, start with a high entropy value
        best_attribute = None
        
        # for each available attribute, calculate the weighted entropy (Enew)
        for attribute in available_attributes:
            weighted_entropy = self.calculate_entropy_of_partition(instances, attribute)
            
            # If this attribute's entropy is smaller than the current minimum, update
            if weighted_entropy < min_entropy:
                min_entropy = weighted_entropy
                best_attribute = attribute
        
        # Return the attribute with the smallest Enew entropy
        return best_attribute
    
    def majority_vote(self, instances, parent_instances = None):
        """ Perform majority voting. If `parent_instances` is provided, use it to determine the total count
        and the majority class; otherwise, use the given `instances`. """

        if parent_instances:
            voting_instances = parent_instances
            total_count = len(voting_instances)
        else:
            voting_instances = instances
            total_count = len(instances)
        
        # Debug: Show which instances are being used for voting
        # print(f"DEBUG: Voting Instances: {voting_instances}")
        # print(f"DEBUG: Total Count for Majority Vote: {total_count}")

        # first need to check if there are instances to do vote
        if not voting_instances:
            return None, total_count
        
        # create a dictionary to count occurence of each class label
        class_counts = {}

        # loop through each instance to count the class label
        for instance in voting_instances:
            label = instance[-1] # class label will be last element

            if label not in class_counts:
                class_counts[label] = 0
            
            class_counts[label] += 1
        
        # print(f"DEBUG: Class Counts: {class_counts}")
        
        # find maximum count among all the class labels
        max_count = max(class_counts.values())

        majority_classes = []

        for label, count in class_counts.items():
            if count == max_count:
                majority_classes.append(label)
        
        # IMPORTANT: if there are ties (multiple classes with same count), sort alphabetically
        # print(f"DEBUG: Majority Classes (before sorting): {majority_classes}")
        majority_classes.sort()
        # print(f"DEBUG: Resolved Majority Class: {majority_classes[0]}")
        return majority_classes[0], len(voting_instances) # return the first class label

    def all_same_class(self, instances):
        """This function checks to see if all instances are members of the same class
    
        Args:
            instances: values
        
        Returns: 
            True or False (if all same class labels)
        """
        first_class = instances[0][-1]
        for instance in instances:
            if instance[-1] != first_class:
                return False
        # get here, then all same class labels
        return True
    
    def tdidt(self, current_instances, available_attributes, parent_instances = None):
        """This function replicates the tdidt algorithm that we use to build our decision tree
    
        Args:
            - current_instances
            - available_attributes
            - parent_instances
        
        Returns: 
            the decision tree built
        """
        split_attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(split_attribute) # can't split on this attribute again

        # To store the index instead of name:
        tree = ["Attribute", self.header.index(split_attribute)]
        att_partition = self.partition_instances(current_instances, split_attribute)

        # for each partition, repeat unless one of the following occurs (base case)
        # print(f"DEBUG: Split on Attribute {split_attribute}, Remaining Attributes: {available_attributes}")
        # print(f"DEBUG: Current Instances: {len(current_instances)}")

        for att_value in sorted(att_partition.keys()): # process in alphabetical order
            # print(f"DEBUG: Attribute Value {att_value}, Partition Size: {len(att_partition[att_value])}")
            att_partition_value = att_partition[att_value]
            value_subtree = ["Value", att_value]

            # Handle CASE 1: All Instances Have the Same Class
            if len(att_partition_value) > 0 and self.all_same_class(att_partition_value):
                # checks if the paritition has attributes and if they are all the same class
                class_label = att_partition_value[0][-1]
                # att_partition[0]: This accesses the first instance (row) in the partition
                # att_partition[0][-1]: The [-1] index accesses the last element of th
                # first instance, which is the class label
                count = len(att_partition_value)  # Count how many instances in this class
                total_count = len(current_instances)
                value_subtree.append(["Leaf", class_label, count, total_count])
            
            # Handle CASE 2: No More Attributes to Split
            elif len(att_partition_value) > 0 and len(available_attributes) == 0:
                # checks if partition has attributes but no more available attributes to split on
                class_label,count = self.majority_vote(att_partition_value, parent_instances)  # call function to do majority vote within instances
                total_count = len(att_partition_value)
                value_subtree.append(["Leaf", class_label, count, total_count])

            # Handle CASE 3: Empty Partition
            # in this case i need to return to the parent partition (so don't append leaf node here!!)
            # recurse back up to the parent partition (current_instances) and perform majority vote
            elif len(att_partition_value) == 0:
                # Perform majority vote using the parent instances (current_instances in this case)
                class_label, count = self.majority_vote(current_instances)
                # Propagate the result upwards without appending a leaf node here
                return ["Leaf", class_label, count, len(parent_instances)]
            else:
                # none of base cases were true, recurse!!
                subtree = self.tdidt(att_partition_value, available_attributes.copy(), parent_instances = current_instances)
                value_subtree.append(subtree)
        
            tree.append(value_subtree)
        return tree


    def fit(self, X_train, y_train):
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

        # programmatically create the header
        self.header = [] # initializes an empty list to store attribute names
        for i in range(len(X_train[0])): # loops over the indexes int he first row of X_train
            self.header.append(f"att{i}") # will append names (e.g. att0, att1)
        
        # programmatically to create attribute domain
        # each key is an attribute name and the value is a list of all
        # unique values that appear in that column of X_train
        self.attribute_domains = {} # initialize an empty dictionary
        for i in range(len(self.header)): # loop through each attribute index
            # collect all unique values for current attribute
            unique_values = []
            for row in X_train:  # loops through each row in the training data
                if row[i] not in unique_values:
                    unique_values.append(row[i])
            self.attribute_domains[self.header[i]] = unique_values

        # first stich together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]

        available_attributes = self.header.copy()

        # call tdidt function here
        self.tree = self.tdidt(train, available_attributes)

        return self.tree

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = [] # will store the returned predictions

        for instance in X_test: # loops through each test instance in X_test
            node = self.tree # start at root of decision tree

            # go through the tree until reach a leaf node (will have prediction)
            while isinstance(node, list): # continues along the tree if node is a list (means its not a leaf node)
                if node[0] == "Leaf":
                    # if the node is a leaf, append the class label to predictions
                    y_predicted.append(node[1]) # format: ["Leaf", class_label, count, total_count]
                    break # exit loop, don't need to go any further
                elif node[0] == "Attribute":
                    # find the attribute to split on - format: ["Attribute", attribute_index, attribute_value]
                    attribute_index = node[1]
                    attribute_value = instance[attribute_index]

                    found = False
                    for partition in node[2:]: # loops over paritions to find subtree that corresponds to current attribute value
                        if partition[0] == "Value" and partition[1] == attribute_value:
                            node = partition[2] # move to the subtree for this value
                            found = True
                            break # exit the loop since correct parition is found
                    
                    # if no matching parition was found, use majority voting
                    if not found:
                        majority_class = self.majority_vote(instance)
                        y_predicted.append(majority_class)
                        break

        return y_predicted


    def print_decision_rules(self, node=None, conditions=None, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            node (list or None): Current node of the decision tree to process.
            conditions (list of str or None): Accumulated conditions (e.g., "att0 == val").
            attribute_names (list of str or None): List of attribute names to use in the decision rules.
            class_name (str): String to use for the class name in the decision rules.
        """
        if node is None:
            node = self.tree  # starts with the root if no node is passed
        
        # this is to track conditions for the the current path in the tree
        if conditions is None:
            conditions = []  # intialize empty conditions if not passed

        # first check to see if we are at a leaf node (base case)
        # format: ["Leaf", class_label ...]
        if node[0] == "Leaf":
             # if leaf start the rule building
            rule = "IF "

            # checks if conditions exist
            if conditions: # if conditions we join the accumulated conditions with " AND "
                rule += " AND ".join(conditions)
            
            # appends the final class assigment (at index 1) to the rule here
            rule += f" THEN {class_name} = {node[1]}"
            print(rule) # prints the completed rule
            return # we print the rule and stop the recursion here in this scenario
 
            
        # recursive case (attribute node)
        # format: ["Attribute", index, [partition1], [partition2]]
        if node[0] == "Attribute":
            # get the index of the attribute being split on (e.g. 0, 1, etc)
            attribute_index = node[1]

            # determine the attribute name for rules
            if attribute_names:
                attribute_name = attribute_names[attribute_index]
            else:
                attribute_name = f"att{attribute_index}"
                
            # need to process each branch/partition in the current attribute node
            # partitions have the format: ["Value", branch_value, subtree]
            for partition in node[2:]:
                    # get the value for this branch
                    value = partition[1] # (e.g. 1, 2, "excellent" etc)
                    subtree = partition[2] # get subtree for this branch (can be another attribute or leaf)

                    condition = f"{attribute_name} == {value}" # constructs the condition string for this branch

                    # create a new list of conditions for this current branch path
                    updated_conditions = conditions[:]
                    updated_conditions.append(condition) # add the new condition

                    # recurse to print rules for the subtree
                    self.print_decision_rules(node = subtree, conditions = updated_conditions, attribute_names = attribute_names, class_name = class_name)



    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
            """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
            its DOT graph language (produces .dot and .pdf files).

            Args:
                dot_fname(str): The name of the .dot output file.
                pdf_fname(str): The name of the .pdf output file generated from the .dot file.
                attribute_names(list of str or None): A list of attribute names to use in the decision rules
                    (None if a list is not provided and the default attribute names based on indexes
                    (e.g. "att0", "att1", ...) should be used).

            Notes:
                Graphviz: https://graphviz.org/
                DOT language: https://graphviz.org/doc/info/lang.html
                You will need to install graphviz in the Docker container as shown in class to complete this method.
            """
            pass # TODO: (BONUS) fix this
