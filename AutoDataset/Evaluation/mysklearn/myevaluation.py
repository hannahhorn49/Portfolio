##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01, Fall 2024
# Programming Assignment #5
# 10/28/24
# I did not attempt the bonus
#
# Description: This program provides the implementation that to pass the tests
#               in test_myevaluation for evaluating classifiers.
##############################################

"""Module providing way for randomness, calculate ceiling function"""
import random
import math
import numpy as np

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Set the random seed after checking if random_state is provided
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle: # randomize the table
        myutils.randomize_in_place(X, parallel_list=y)
   
    # get total length to calculate test size
    total_length = len(X)
  
    # check if test_size is a int or float
    if isinstance(test_size, int):
        test_length = test_size
    elif isinstance(test_size, float):
        test_length = math.ceil(test_size * total_length)
    else:
        raise ValueError("Invalid test_size input. Must be a float or int.")
    
        # Ensure the number of test samples does not exceed total samples
    if test_length > total_length:
        raise ValueError("Test size cannot be larger than the number of samples.")
    
    # Calculate training size
    train_size = len(X) - test_length

    # Initialize training and testing sets
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # Populate the training sets
    for i in range(train_size):
        X_train.append(X[i])
        y_train.append(y[i])

    # Populate the testing sets
    for j in range(train_size, len(X)):
        X_test.append(X[j])
        y_test.append(y[j])
        
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    # find list of indexes
    total_samples = len(X)

    # check to make sure we don't split more than total samples
    if n_splits > total_samples:
        raise ValueError(f"Number of splits ({n_splits}) cannot be greater than the number of samples ({total_samples})")

    indexes = list(range(total_samples))
    
    if shuffle: # randomize the indexes
        np.random.shuffle(indexes)
    
    # this will initialize folds with n splits empty sublists
    folds = []
    for _ in range(n_splits):
        folds.append([])
    
    for i, index in enumerate(indexes):
        fold_index = i % n_splits
        folds[fold_index].append(index)

    # final list to hold each (train_indices, test_indices) tuple
    folds_with_train_test = []

    for i in range(n_splits):
        test_indices = folds[i] # first iteration: i = 0, [0,2 test set]
        train_indices = [] # empty list to store training samples
        
        for j in range(n_splits): # add samples to train indexes by looping over all folds (except current fold.. ex. i= 0)
            if j != i: # j = 0, matches i so skip, don't add test samples to training indexes
                for idx in folds[j]:
                    train_indices.append(idx)

        folds_with_train_test.append((train_indices, test_indices)) # for first iteration, adding tuple: ([1,3], [0,2])

    return folds_with_train_test

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    return [] # TODO: (BONUS) fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """

    # sets the random seed after checking if random_state is provided
    if random_state:
        random.seed(random_state)
    else:
        random.seed(49)
    # determine the number of samples (n_samples) if not provided
    if n_samples is None:
        n_samples = len(X)
    # initialize empty lists
    X_sample = [] # stores bootstrapped training samples
    X_out_of_bag = [] # stores out-of-bag test samples
    y_sample = [] # stores bootstrapped target values (if y is not none)
    y_out_of_bag = [] # stores out-of-bag target values (if y is not none)

    # need a set to track the unique indexes that were selected
    selected_indexes = set()

    # use loop to generate the bootstrap sample (training and target) with replacement
    for _ in range(n_samples):
        index = random.randint(0, len(X) - 1)
        X_sample.append(X[index])
        if y is not None:
            y_sample.append(y[index])
        selected_indexes.add(index)

    # use loop to create out-of-bag samples
    for i in range(len(X)):
        if i not in selected_indexes:
            X_out_of_bag.append(X[i])
            if y is not None:
                y_out_of_bag.append(y[i])

    if y is not None:
        return X_sample, X_out_of_bag, y_sample, y_out_of_bag
    else:
        return X_sample, X_out_of_bag, None, None


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # good practice to ensure that the lengths of y_true and y_pred match
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same!")
    total_labels = len(labels) # explain why using len here?

    matrix = [] # this is eventually where the confusion matrix will go

    # first fill out confusion matrix outline with 0s for each part
    for i in range(total_labels):
        row = []
        for j in range(total_labels):
            row.append(0)  # Initialize each entry to zero
        matrix.append(row)  # Add the row to the matrix

    label_to_index = {}
    for index in range(total_labels):
        label_to_index[labels[index]] = index

    # need to loop through a populate the matrix
    for i in range(len(y_true)):
        true_label = y_true[i]  # gets the true label at the index
        predicted_label = y_pred[i]  # gets the predicted label at that index

        true_index = label_to_index[true_label]
        predicted_index =  label_to_index[predicted_label]

        matrix[true_index][predicted_index] += 1 # add one count to that part of matrix

    return matrix



def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
            score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """

    # good practice to ensure that the lengths of y_true and y_pred match
    if len(y_true) != len(y_pred):
        raise ValueError("The lengths of y_true and y_pred must be the same!")

    # need to keep track of how many correct predictions when loop through
    correct_predictions = 0

    # loop through true values and corresponding predictions
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]: # check to see if there's a match
            correct_predictions += 1

    # need to provide implementation if normalize is True
    if normalize is True:
        return correct_predictions / len(y_true)
    else:
        return correct_predictions
