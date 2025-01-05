##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01, Fall 2024
# Programming Assignment #5
# 10/28/24
# I did not attempt the bonus
# 
# Description: This program contains the general reusuable utilty functions
#               that will be used in the jupyter notebook.
##############################################

import numpy as np # use numpy's random number generation
from mysklearn.mypytable import MyPyTable
from mysklearn import myevaluation
from tabulate import tabulate


def compute_euclidean_distance(v1, v2):
    """This function computes the euclidean distance between two instances
    Args: two instances

    Returns: the euclidean distance (float)

    """
    return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))

def doe_rating_assign(mpg):
    """this function is for the Discretization and assigns mpg certain DOE ratings
    Args: mpg(double) value

    Returns: the corresponding DOE rating for that mpg value    
    """
    mpg = round(mpg)
    if mpg >= 45:
        return 10
    elif 37 <= mpg <= 44:
        return 9
    elif 31 <= mpg <= 36:
        return 8
    elif 27 <= mpg <= 30:
        return 7
    elif 24 <= mpg <= 26:
        return 6
    elif 20 <= mpg <= 23:
        return 5
    elif 17 <= mpg <= 19:
        return 4
    elif 15 <= mpg <= 16:
        return 3
    elif mpg == 14:
        return 2
    elif mpg <= 13:
        return 1
    else:
        return 0


def normalize_train_attribute(column_values):
    """Normalizes a list of values to the range [0, 1].
    Args: column values (list)

    Returns: the normalized attributes in a list
    
    """
    min_value = min(column_values)
    max_value = max(column_values)
    range_value = max_value - min_value

    normalized_attribute = []
    for value in column_values:
        new_value = (value - min_value)/range_value
        normalized_attribute.append(new_value)
    return normalized_attribute


def combine_normalized_attributes(normalized_cylinder, normalized_weight, normalized_acceleration):
    """ Combines the normalized attributes into a 2D list
        Args: the normalized values (lists)

        Returns: combined instances (list)

    """
    combined_instances = []
    for i in range(len(normalized_cylinder)):  # Assuming all lists have the same length
        row = [normalized_cylinder[i], normalized_weight[i], normalized_acceleration[i]]
        combined_instances.append(row)
    return combined_instances

def randomize_in_place(alist, parallel_list=None):
    """ This function randomizes a list in place (optional parallel list)

        Args: alist (list)
        Returns: nothing, shuffles list in place

    """
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] = parallel_list[rand_index], parallel_list[i]

def prepare_data_random_subsamples():
    """ This function prepares the data by loading the dataset, getting the relevant columns,
        normalizing the features, and preparing the data as split X_train and y_train

        Args: None
        Returns: 
            combined_x_train (2D list of normalized features)
            y_train (list) of target mpg ratings
    """
    dataset = MyPyTable()
    dataset.load_from_file("auto-data-remove-NA.txt")

    # get columns 
    cylinder_values = dataset.get_column("cylinders")
    weight_values = dataset.get_column("weight")
    acceleration_values = dataset.get_column("acceleration")

    # normalize each feature
    normalized_cylinder = normalize_train_attribute(cylinder_values)
    normalized_weight = normalize_train_attribute(weight_values)
    normalized_acceleration = normalize_train_attribute(acceleration_values)

    # combine normalized attributes into a 2D list
    combined_x_train = combine_normalized_attributes(normalized_cylinder, normalized_weight, normalized_acceleration)

    # extract target (y_value) labels (DOE mpg ratings) that correspond to each instance
    mpg_values = dataset.get_column("mpg")
    y_train = []
    for value in mpg_values:
        rating = doe_rating_assign(value)
        y_train.append(rating)
    
    return combined_x_train, y_train


def random_subsample(X, y, classifier, k=10, test_size=0.33, random_state=None):
    """Performs random subsampling for evaluating a classifier.

    Args:
        X (list of list of obj): Features of the dataset.
        y (list of obj): Target labels of the dataset.
        classifier: An instance of the classifier with fit and predict methods.
        k (int): Number of subsampling iterations.
        test_size (float): Proportion of dataset to include in the test split.
        random_state (int): Seed for random number generator for reproducibility.

    Returns:
        average_accuracy (float): Average accuracy over k subsampling iterations.
        average_error_rate (float): Average error rate over k subsampling iterations.
    """

    accuracies = []
    # Initialize random_seed as None before the loop
    random_seed = None

    for i in range(k):
        # Update random_seed only if random_state is provided
        if random_state is not None:
            random_seed = random_state + i

        # Split the data
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=test_size, random_state=random_seed)

        # Train the classifier on the training set
        classifier.fit(X_train, y_train)

        # Predict on the test set
        predictions = classifier.predict(X_test)

        # Calculate accuracy and error rate
        accuracy = myevaluation.accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    # Compute average accuracy and error rate
    average_accuracy = np.mean(accuracies)
    average_error_rate = 1 - average_accuracy

    return average_accuracy, average_error_rate

def cross_val_predict(X, y, classifier, n_splits=10, random_state=None, shuffle=False):
    """Generates cross-validated predictions from the input classifier.

    Args:
        X (list of list of obj): Features of the dataset.
        y (list of obj): Target labels of the dataset.
        classifier: An instance of the classifier with fit and predict methods.
        n_splits (int): Number of folds for cross-validation.
        random_state (int): Seed for random number generator for reproducibility.
        shuffle (bool): Whether to shuffle the data before splitting.

    Returns:
        predictions (list of obj): Predicted labels for each instance in the dataset.
    """

    predictions = [None] * len(y)

    # call kfold_split to get number of splits
    folds = myevaluation.kfold_split(X, n_splits=n_splits, random_state=random_state, shuffle=shuffle)

    for train_indexes, test_indexes in folds:
        # intialize lists to hold training and testing data for the current fold
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for i in range(len(X)):
            if i in train_indexes:
                X_train.append(X[i])
                y_train.append(y[i])
            elif i in test_indexes:
                X_test.append(X[i])
                y_test.append(y[i])
        
        # fit classifier
        classifier.fit(X_train, y_train)

        # call predict method and store predictions
        # predicts the labels for the current test set
        fold_predictions = classifier.predict(X_test)

        # Loop over each index in the test_indices list
        for index in range(len(test_indexes)):
            # store current test index
            current_test_index = test_indexes[index]
            
            # get the corresponding prediction for this test index from fold_predictions
            current_prediction = fold_predictions[index]
            
            # place the prediction in the predictions list at the position of current_test_index
            predictions[current_test_index] = current_prediction


    return predictions

def bootstrap_method(X, y, classifier, k =10, random_state = None):
    """Perform bootstrap resampling to evaluate classifier performance.

    Args:
        classifier: The classifier object that has fit and predict methods.
        X (list of list of obj): The feature data.
        y (list of obj): The target values.
        k (int): The number of bootstrap samples to generate (default is 10).
        random_state (int): Seed for reproducibility.

    Returns:
        avg_accuracy (float): The average predictive accuracy across all bootstrap samples.
        avg_error_rate (float): The average error rate across all bootstrap samples.
    """

    # stores the accuracy and error rate for each bootstrap sample
    accuracies = []
    error_rates = []

    for i in range(k):
        # call function 
        X_sample, X_out_of_bag, y_sample, y_out_of_bag = myevaluation.bootstrap_sample(X, y, random_state = random_state)

        # train classifier 
        classifier.fit(X_sample, y_sample)

        # call predict method on the out_of_bag samples (aka testing set "unseen")
        y_predictions = classifier.predict(X_out_of_bag)

        # need to calculate the number of correct predictions made by classifier
        # compare against y_out_of_bag
        correct_predictions = 0

        # use zip to interate over the true and predicted labels
        for actual, prediction in zip(y_out_of_bag, y_predictions):
            # if the actual label matches predicted label, add one to correct predictions
            if actual == prediction:
                correct_predictions += 1
        
        # calculate the accuracy and error rate
        accuracy = correct_predictions / len(y_out_of_bag)
        error_rate = 1 - accuracy

        # store this accuracy and error rate to the lists
        accuracies.append(accuracy)
        error_rates.append(error_rate)

    # calculate average accuracy and error rates over all bootstrap samples
    average_accuracy = sum(accuracies) / len(accuracies)
    average_error_rate = sum(error_rates) / len(error_rates)

    return average_accuracy, average_error_rate

def print_confusion_matrix_with_metrics(confusion_matrix, labels):
    """
    Prints a confusion matrix with MPG Ranking, Total, and Recognition (%) columns.

    Parameters:
    - confusion_matrix (list of lists): The original confusion matrix (2D list).
    - labels (list of str): The headers for each class.
    """
    # eventually hold each row of the confusion matrix along with the total and recognition percentage
    matrix_with_metrics = []
    
    # loop through each row and its index in the confusion matrix
    for i, row in enumerate(confusion_matrix):
        # calculates total for the current row
        row_total = sum(row)
        
        # calculates recognition rate
        if row_total > 0:
            recognition = (row[i] / row_total) * 100
        else:
            recognition = 0

        # add onto existing row in confusion matrix the row total and recognition
        row_with_metrics = row + [row_total, recognition]
        
        # append this new row to the list
        matrix_with_metrics.append(row_with_metrics)
    
        # define the MPG rankings to use as row labels
        mpg_ranking = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # update headers to include "MPG Ranking", "Total", and "Recognition (%)"
    headers = ["MPG Ranking"] + labels + ["Total", "Recognition (%)"]

    # print the confusion matrix with the additional metrics
    print(tabulate(matrix_with_metrics, headers=headers, showindex=mpg_ranking, floatfmt=".1f"))
