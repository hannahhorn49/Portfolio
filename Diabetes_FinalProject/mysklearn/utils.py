##############################################
# Programmer: Hannah Horn, Eva Ulrichsen
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #final project
# 12/9/24
# I did not attempt the bonus
# Description: This program contains helper functions for classifiers.
# Contains math functions and general use functions.
#########################

import numpy as np
from tabulate import tabulate
from mysklearn.mypytable import MyPyTable
from mysklearn import myevaluation
import math

def get_column(table, header, col_name):
    """Extracts a column from the table data as a list.

        Args:
            col_name(str or int): string for a column name or int
                for a column index
            table(list of obj): 2D data structure 
            header(list of str): M column names

        Returns:
            list of obj: 1D list of values in the column
        """
    col_index = header.index(col_name)
    col = []
    for row in table:
        col.append(row[col_index])

    return col

def compute_equal_width_cutoffs(values, num_bins):
    """Computes the cutoff values for discretization.

        Args:
            values(list of int): 1D list of data in a column
            num_bins(int): int for M bins 

        Returns:
            list of float: 1D list of calculated bin values
        """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    cutoffs = [min(values) + i * bin_width for i in range(num_bins)]
    cutoffs.append(max(values))
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]

    return cutoffs

def get_frequencies(table, header, col_name):
    """Get the frequencies of values in column

        Args:
            col_name(str or int): string for a column name or int
                for a column index
            table(list of obj): 2D data structure 
            header(list of str): M column names

        Returns:
            list of obj: 1D list of values in the column
            list of obj: 1D list of frequency of values
        """
    col = get_column(table, header, col_name)

    frequency_dict = {}

    for val in col:
        if val in frequency_dict:
            frequency_dict[val] += 1
        else:
            frequency_dict[val] = 1

    unique_col_values = sorted(frequency_dict.keys())
    counts = [frequency_dict[val] for val in unique_col_values]

    return unique_col_values, counts


def compute_bin_frequencies(values, cutoffs):
    """Find frequencies of values at defined cutoffs

        Args:
            values(list of obj): 1D list of int or str
            cutoffs(list of obj): 1D list of int or string

        Returns:
            list of obj: 1D list of frequencies of values at cutoffs
        """
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for value in values:
        if value == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def compute_slope_intercept(x, y):
    """Computes the slope and y-intercept

        Args:
            x(list of int): 1D list of int 
            y(list of int): 1D list of int

        Returns:
            m(int): slope
            b(int): y-intercept
        """
    meanx = sum(x) / len(x)
    meany = sum(y) / len(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den
    b = meany - m * meanx

    return m, b

def calculate_r_value(x, y):
    """Calculate r coefficient

        Args:
            x(list of int): 1D list of int
            y(list of int): 1D list of int

        Returns:
            r(int): value computed between [-1, 0]
        """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (denominator_x * denominator_y) ** 0.5

    if denominator == 0:
        return 0

    r = numerator / denominator
    return r

def class_discretizer(prediction):
    """
    Discretizes the list of predicted MPG values into DOE mpg ratings.

    Args:
        predictions (list of float): A list of predicted mpg values.

    Returns:
        ratings (list of int): A list of corresponding DOE mpg ratings.
    """
    ranges = [0, 14, 15, 17, 20, 24, 27, 31, 37, 45, 46]
    ratings = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    y_predicted = []

    for value in prediction:
        rating = None

        for i in range(len(ranges) - 1):
            if ranges[i] <= value < ranges[i + 1]:
                rating = ratings[i]
                break

        if rating is None:
            if value >= 45:
                rating = 10
            elif value <= 13:
                rating = 1

        y_predicted.append(rating)

    return y_predicted

def compute_euclidean_distance(x, y):
    """Computes the Euclidean distance between x and y.
    altered for categorical vals

    Args:
        x (list of int or float): first value
        y (list of int or float): second value

    Returns:
        dists: The Euclidean distance between vectors x and y.     
    """
    distances = []
    for index, value in enumerate(x):
        if isinstance(value, (int, float)):
            distances.append((value - y[index]) ** 2)
        elif value == y[index]:
            distances.append(0)
        else:
            distances.append(1)

    return np.sqrt(sum(distances))

def min_max_scale(column):
    """Scales a list of numeric values using min-max scaling.
    
    Args: 
        column (1D list): numeric values in a list
    
    Returns: 
        values: the scaled values of the values in column

    """
    min_val = min(column)
    max_val = max(column)

    if max_val == min_val:
        return [0 for _ in column]

    return [(x - min_val) / (max_val - min_val) for x in column]

def clean_and_save_data(input_file, output_file):
    """Load data, remove rows with 'no info' in the smoking history, and save the cleaned data."""

    # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file(input_file)

    # check the shape of the data before cleaning
    before_drop_shape = table.get_shape()
    print(f"Before cleaning - Rows: {before_drop_shape[0]}, Columns: {before_drop_shape[1]}")

    # extract the smoking history column and find rows to drop
    smoking_column = table.get_column("smoking_history")

    rows_to_drop = []
    for index, value in enumerate(smoking_column):
        if value == "No Info":
            rows_to_drop.append(index)

    # drop the rows with "No Info" in the smoking history column
    table.drop_rows(rows_to_drop)

    # check the shape of the data after cleaning
    after_drop_shape = table.get_shape()
    print(f"After cleaning - Rows: {after_drop_shape[0]}, Columns: {after_drop_shape[1]}")

    # save the cleaned data to a new file
    table.save_to_file(output_file)

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

def prepare_categorical_data_random_subsamples():
        """ This function prepares the data by loading the dataset, getting the relevant columns,
        and preparing the data as split X_train and y_train

        Args: None
        Returns: 
            combined_x_train (2D list of normalized features)
            y_train (list) of target mpg ratings
        """

        dataset = MyPyTable()
        dataset.load_from_file("input_data/titanic.csv")

        # get columns
        class_values = dataset.get_column("class")
        age_values = dataset.get_column("age")
        sex_values = dataset.get_column("sex")

        # combine these values into the X
        combined_instances = []
        for i in range(len(class_values)):  # Assuming all lists have the same length
            row = [class_values[i], age_values[i], sex_values[i]]
            combined_instances.append(row)
        
        survived_values = dataset.get_column("survived")

        return combined_instances, survived_values


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
    folds = myevaluation.stratified_kfold_split(X, y, n_splits=n_splits, random_state=random_state, shuffle=shuffle)

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

def evaluate_classifier(X, y, classifier, pos_label = 1.0, n_splits = 10):
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
    predictions = cross_val_predict(X, y, classifier, n_splits=n_splits, random_state=1, shuffle=True)
    
    # calculate metrics
    accuracy = myevaluation.accuracy_score(y, predictions)
    error_rate = 1 - accuracy
    precision = myevaluation.binary_precision_score(y, predictions, pos_label=pos_label)
    recall = myevaluation.binary_recall_score(y, predictions, pos_label=pos_label)
    f1 = myevaluation.binary_f1_score(y, predictions, pos_label=pos_label)
    
    # create the general confusion matrix
    labels = [0.0, pos_label]
    confusion = myevaluation.confusion_matrix(y, predictions, labels)
    
    # display results
    print(f"Accuracy = {accuracy:.2f}, error rate = {error_rate:.2f}")
    print(f"Precision = {precision:.2f}")
    print(f"Recall = {recall:.2f}")
    print(f"F1 Measure = {f1:.2f}")
    print("\nConfusion Matrix:")
    print_confusion_categorical_matrix_with_metrics(confusion, labels)

def combine_normalized_attributes2(*columns):
    """ Combines normalized and categorical attributes into a 2D list.
    
    Args:
        *columns: Any number of lists representing normalized and categorical attributes.
    
    Returns:
        list: A 2D list of combined instances.
    """
    # Ensure all columns have the same length
    num_rows = len(columns[0])  # Assume all columns have the same length
    for col in columns:
        if len(col) != num_rows:
            raise ValueError("All columns must have the same length.")
    
    combined_instances = []
    for i in range(num_rows):
        row = []
        for column in columns:
            row.append(column[i])
        combined_instances.append(row)
    
    return combined_instances

def prepare_mixed_data():
   data = MyPyTable()
   data.load_from_file("output_data/diabetes_minimize.csv")
   
   # get numeric X columns and normalize
   age_values = data.get_column("age")
   a1c_values = data.get_column("HbA1c_level")
   glucose_values = data.get_column("blood_glucose_level")
   
   normalized_age = normalize_train_attribute(age_values)
   normalized_a1c = normalize_train_attribute(a1c_values)
   normalized_glucose_level = normalize_train_attribute(glucose_values)
   
   # get rest of X columns
#    gender_values = data.get_column("gender")
   hyptertension_values = data.get_column("hypertension")
   heart_disease_values = data.get_column("heart_disease")
#    smoking_values = data.get_column("smoking_history")
   
   # combine these values into X
   X = combine_normalized_attributes2(normalized_age, normalized_a1c, normalized_glucose_level, hyptertension_values, heart_disease_values)
   
   # get target value
   diabetes = data.get_column("diabetes")
   
   return X, diabetes

def calculate_entropy(instances, attributes):
    # fix this
    """ Calculates entropy for a given attribute of target values. """
    entropies = []
    class_labels = [instance[-1] for instance in instances]
    unique_class_labels = list(list(np.unique(class_labels)))
    # print("unique class labels:", unique_class_labels)

    for attribute in attributes:
        att_entropies = []
        unique_att_counts = []
        att_index = int(attribute[-1])
        att_values = [instance[att_index] for instance in instances]
        unique_att_vals, unique_att_counts = np.unique(att_values, return_counts=True)

        # Calculate entropy
        for index, value in enumerate(unique_att_vals):
            class_counts = []
            # print("value:", value)
            for i in range(len(unique_class_labels)):
                class_counts.append(0)
            #print("len of class counts: ", len(class_counts))
            for instance in instances:
                if instance[att_index] == value:
                    #print("index ", class_labels.index(instance[-1]))
                    class_counts[unique_class_labels.index(instance[-1])] += 1
            # print("class counts:", class_counts)
            entropy = 0
            if 0 not in class_counts:
                for count in class_counts:
                    probability = count / unique_att_counts[index]
                    # print("probability:", probability)
                    entropy -= probability * (math.log2(probability)) # if probability > 0 else 0
            att_entropies.append(entropy)
        # print("att entropies:", att_entropies)
        att_entropy = 0

        # Calculate Enew
        for index, entropy_val in enumerate(att_entropies):
            att_entropy += (unique_att_counts[index] / len(instances)) * entropy_val
        entropies.append(att_entropy)

        #print(f"For attribute index {attribute}, entropy: {att_entropy}")
    return entropies

def partition_instances(header, instances, attribute, attribute_domains):
    """ Helper function that partitions instances by attribute domain.

    Args:
        header (list of str): used to find att index
        instances (2D list): data used in partitions
        attribute (int or str): what is being split on
        attribute_domains (dictionary): all of the groups

    Returns:
        partitions (dictionary): instances divided by attribute
    
    """
    # this is group by attribute domain (not values of attribute in instances)
    # lets use dictionaries
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    # print("att domain in partition: ", att_domain)
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def all_same_class(instances):
    """ Determines if all same instances belong to same class.

    Args:
        instances (list): values to be checked

    Returns:
        true/false (boolean): if condition is met
    """
    first_class = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_class:
            return False
    # get here, then all same class labels
    return True

def majority_vote(instances, previous_instances=None):
    """ Returns the majority class label.

    Args:
        instances (list): data
        previous_instances (list): data from prev partition

    Returns:
        majority_label (str): class label
        len of instances (int): number of instances to count    
    """
    if previous_instances:
        total_instances = previous_instances
    else:
        total_instances = instances
    # print("previous instances:", total_instances)
    class_labels = [instance[-1] for instance in total_instances]
    label_counts = {}

    # Count occurrences of each label
    for label in class_labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    # Find the maximum count and resolve clashes
    max_count = max(label_counts.values())
    majority_labels = [label for label, count in label_counts.items() if count == max_count]

    if len(majority_labels) > 1:
        # print("min majority label: ", min(majority_labels))
        return min(majority_labels), len(total_instances)  # Return the first in alphabetical order

    # print("majority label:", majority_labels[0])

    return majority_labels[0], len(total_instances)
