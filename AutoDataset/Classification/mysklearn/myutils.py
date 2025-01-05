"""Module allows us to seed the random numbers and have access to MyPyTable methods"""
import random
import numpy as np
from mypytable import MyPyTable

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

def normalize_test_attribute(column_values, x_test_indexes):
    """Normalizes a list of values to the range [0, 1].
    Args: column values (list), x_test_indexes (integers)

    Returns: the normalized attribute in a list

    """
    min_value = min(column_values)
    max_value = max(column_values)
    range_value = max_value - min_value

    normalized_attribute = []
    for index in x_test_indexes:
        value = column_values[index]
        new_value = (value-min_value)/range_value
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

def select_random_instances_from_dataset(dataset, k):
    """Selects k random instances from a dataset.

    Args:
        dataset: A list of data instances.
        k: The number of instances to select.

    Returns:
        A list of k randomly selected instances.
    """
    # seed random number generator
    random.seed(147)
    # generates the random indexes we use from data
    indexes = random.sample(range(len(dataset)), k)
    return indexes

def load_and_prepare_train_data(filename):
    """ This function loads the file and drops the test indexes to create
        the training set.

        Args: filename: the name of the file

        Returns: the new table without the test indexes (training data)
    
    """
    table = MyPyTable()
    table.load_from_file(filename)

    # Select random instances
    test_indices = select_random_instances_from_dataset(table.data, 5)
    table.drop_rows(test_indices)  # Drop the test rows immediately

    return table

def load_and_prepare_test_data(filename):
    """ This function loads a file into a new MyPyTable object

    Args: filename: name of the file

    Returns: a new table with all the original data
    
    """
    original_table = MyPyTable()
    original_table.load_from_file(filename)
    return original_table

def calculate_accuracy(model_predictions, actual_ratings, x_test_indexes):
    """ This function calculates the accuracy of a classifier.

        Args: 
            model predictions: list of predictions made by the model
            actual_ratings: list of actual ratings of test indexes
            x_test_indexes: list of integer indexes for testing set
        
        Returns: The number of correct predictions and the overall accuracy percentage
    
    """

    correct_predictions = 0
    for i, prediction in enumerate(model_predictions):
        if prediction == actual_ratings[i]:
            correct_predictions += 1
    print("This is the number of correct predictions:", correct_predictions)

    accuracy_percentage = (correct_predictions / len(x_test_indexes)) * 100
    print(f"This is the accuracy of the classifier based on the test instances, {accuracy_percentage:.1f}%")