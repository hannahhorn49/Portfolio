##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #7
# 11/20/24
# I did not attempt the bonus
# Description: This program contains various reusable
# functions that I use in the Jupyter Notebook.
##############################################

import numpy as np # use numpy's random number generation
from tabulate import tabulate
from mysklearn.mypytable import MyPyTable
from mysklearn import myevaluation

def generate_training_data(seed, num_samples, slope, intercept, noise_std):
    """Generates the training data 
    Args: 
        seed (int): random seed number
        num_samples: how many random samples
        slope:
        intercept:
        noise_std:
    
    Returns: 
        X_train: list of values for X_train
        y_train: list of y_values (parallel to X_train)
    
    """
    np.random.seed(seed)
    x_train = []
    y_train = []
    for _ in range(num_samples):
        x = np.random.randint(0, 150)
        noise = np.random.normal(0, noise_std)
        y = slope * x + intercept + noise
        x_train.append([x])
        y_train.append(y)
    return x_train, y_train

def discretizer(value):
    """Discretizes a single numeric value into 'high' or 'low'.
    Args: value 

    Returns: a discretized value (either high or low)
    
    """
    if value >= 100:
        return "high"
    else:
        return "low"

def compute_mixed_euclidean_distance(v1, v2, categorical):
    """Compute the mixed Euclidean distance between two instances with both categorical and numerical attributes.

    Args:
        v1 (list): First instance.
        v2 (list): Second instance.
        categorical (list of bool): List indicating if each feature is categorical (True) or numerical (False).

    Returns:
        float: The mixed Euclidean distance between the two instances.
    """

    # check to make sure that when the categorical list is empty, handled properly
    if not categorical:
        categorical = [False] * len(v1)

    distance = 0
    for i in range(len(v1)):
        if categorical[i]:  # if the attribute is categorical
            if v1[i] != v2[i]:
                distance += 1
        else:  # if the attribute is numerical, add the squared difference
            difference = v1[i] - v2[i]
            distance += difference ** 2

    return distance ** 0.5

def prepare_categorical_data_random_subsamples():
        """ This function prepares the data by loading the dataset, getting the relevant columns,
        and preparing the data as split X_train and y_train

        Args: None
        Returns: 
            combined_x_train (2D list of normalized features)
            y_train (list) of target mpg ratings
        """

        dataset = MyPyTable()
        dataset.load_from_file("input_data/tournament_games2016-2021.csv")
        dataset.get_shape()
        dataset.drop_columns([0, 1, 2])
        dataset.get_shape()

        # get column(s)
        tournmaent_seed = dataset.get_column("TournamentSeed")
        X_2D = []
        for value in tournmaent_seed:
            X_2D.append([value]) # the decision tree classifier expects a 2D list

        winner = dataset.get_column("Winner")

        return X_2D, winner

def prepare_categorical_data_with_selected_columns(columns_to_select):
    """This function prepares the data by loading the dataset, extracting the specified columns for X_train,
    and using a specific column for y_train.

    Args:
        columns_to_select (list of str): List of column names to be used for features (X_train).
        
    Returns:
        X_train (2D list of features): The feature set.
        y_train (list): The target values.
    """
    dataset = MyPyTable()
    dataset.load_from_file("input_data/tournament_games2016-2021.csv")
    dataset.get_shape()

    # Drop unnecessary columns (adjust this based on your dataset)
    dataset.drop_columns([0, 1, 2])  # Modify based on which columns are unnecessary

    # Prepare y_train (target variable)
    y_train = dataset.get_column("Winner")  # Assuming 'Winner' is the target variable

    # Prepare X_train (features)
    X_train = []

    # Create an empty list for selected column data
    selected_columns_data = []
    
    # Manually iterate over the columns in columns_to_select
    for column in columns_to_select:
        column_data = dataset.get_column(column)
        selected_columns_data.append(column_data)

    # Iterate over rows of the dataset (assuming each row is an instance/sample)
    for i in range(len(selected_columns_data[0])):  # Loop through each instance
        row = []
        # Iterate through the data of selected columns
        for column_data in selected_columns_data:
            row.append(column_data[i])  # Append the value of the feature for the current row
        X_train.append(row)  # Add the row (feature values for one instance) to X_train

    return X_train, y_train



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

def print_confusion_categorical_matrix_with_metrics(confusion_matrix, labels):
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

        # Add the "Actual Class" column at the beginning of each row
        row_with_metrics = [labels[i]] + row + [row_total, recognition]
        
        # append this new row to the list
        matrix_with_metrics.append(row_with_metrics)
    
    
    # update headers to include "MPG Ranking", "Total", and "Recognition (%)"
    headers = ["Actual Class"] + labels + ["Total", "Recognition (%)"]

    # print the confusion matrix with the additional metrics
    print(tabulate(matrix_with_metrics, headers=headers, floatfmt=".1f"))

def evaluate_classifier(X, y, classifier, pos_label = "H", n_splits = 10):
    """ This function will return the evaluation scores for various metrics for 
        each classifier.

        Args: 
        X (list of list of obj): The feature data.
        y (list of obj): The target values.
        classifier: The classifier object that has fit and predict methods.
        pos_label = the positive class label
        n_splits (int): Number of folds for cross-validation.

        Returns: the print output

    """
    predictions = cross_val_predict(X, y, classifier, n_splits=n_splits)
    
    # calculate metrics
    accuracy = myevaluation.accuracy_score(y, predictions)
    error_rate = 1 - accuracy
    precision = myevaluation.binary_precision_score(y, predictions, pos_label=pos_label)
    recall = myevaluation.binary_recall_score(y, predictions, pos_label=pos_label)
    f1 = myevaluation.binary_f1_score(y, predictions, pos_label=pos_label)
    
    # create the general confusion matrix
    labels = sorted(set(y))
    confusion = myevaluation.confusion_matrix(y, predictions, labels)
    
    # display results
    print(f"Accuracy = {accuracy:.2f}, error rate = {error_rate:.2f}")
    print(f"Precision = {precision:.2f}")
    print(f"Recall = {recall:.2f}")
    print(f"F1 Measure = {f1:.2f}")
    print("\nConfusion Matrix:")
    print_confusion_categorical_matrix_with_metrics(confusion, labels)



def evaluate_tree_classifier(classifier, pos_label = "H"):
    """ This function will return the evaluation scores for various metrics for 
        each classifier.

        Args: 
        classifier: The classifier object that has fit and predict methods.
        pos_label = the positive class label

        Returns: the print output

    """
    columns_to_select = ["RegularSeasonFGPercentMean", "RegularSeasonFG3PercentMean", "LastOrdinalRank"]
    X_train, y_train = prepare_categorical_data_with_selected_columns(columns_to_select)


    # instead of cross_val_predict want to train classifier on entire dataset
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_train)

    # calculate metrics
    accuracy = myevaluation.accuracy_score(y_train, predictions)
    error_rate = 1 - accuracy
    precision = myevaluation.binary_precision_score(y_train, predictions, pos_label=pos_label)
    recall = myevaluation.binary_recall_score(y_train, predictions, pos_label=pos_label)
    f1 = myevaluation.binary_f1_score(y_train, predictions, pos_label=pos_label)
        
    # create the general confusion matrix
    labels = sorted(set(y_train))
    confusion = myevaluation.confusion_matrix(y_train, predictions, labels)
        
    # display results
    print("========================================================")
    print("Decision Tree Results- Trained Over Whole Dataset")
    print("========================================================")
    print(f"Accuracy = {accuracy:.2f}, error rate = {error_rate:.2f}")
    print(f"Precision = {precision:.2f}")
    print(f"Recall = {recall:.2f}")
    print(f"F1 Measure = {f1:.2f}")
    print("\nConfusion Matrix:")
    print_confusion_categorical_matrix_with_metrics(confusion, labels)

