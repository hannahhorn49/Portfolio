import matplotlib.pyplot as plt
from mypytable import MyPyTable

# THESE FUNCTIONS ARE USED TO CALCULATE LINEAR REGRESSION LINE/R-CORRELATION COEFFICIENT

def calculate_mean(x, y):
    """ this function calculates mean of x and y """
    x_mean = sum(x) / len(x)  
    y_mean = sum(y) / len(y) 
    return x_mean, y_mean

def find_difference(x, x_mean, y, y_mean):
    """ this function subtracts the mean from the individual data points
        and stores the difference in a list"""
    # create empty lists to store the differences
    x_diff = []
    y_diff = []

    # subtract the mean from each value in x and append to x_diff
    for val in x:
        x_diff.append(val - x_mean)

    # subtract the mean from each value in y and append to y_diff
    for val in y:
        y_diff.append(val - y_mean)

    return x_diff, y_diff

def calculate_numerator(x_diff, y_diff):
    """ this function will multiply the corresponding x_diff and y_diff values 
        for each value in the list"""
    numerator = 0  # initialize the numerator variab;e

    for i in range(len(x_diff)):
        numerator += x_diff[i] * y_diff[i] 

    return numerator

def calculate_xdiff_squared(x_diff):
    """ this function will calculate the denominator 
        for the slope function by squaring x_diff"""
    x_diff2 = 0  # initialize the denominator to 0

    # loop through each value in x_diff, square it, then add it to variable (summation)
    for value in x_diff:
        x_diff2 += value ** 2 

    return x_diff2

def calculate_ydiff_squared(y_diff):
    """ this function calculates the part of the denominator needed
        for the correlation coefficient"""
    y_diff2 = 0  # initialize the denominator to 0

    # same process as x_diff
    for value in y_diff:
        y_diff2 += value ** 2 

    return y_diff2

def calculate_r_denominator(x_diff2, y_diff2):
    """ this calculates the last part of the denominator
        which is taking square root"""
    denominator = ((x_diff2 * y_diff2) ** (1/2))
    return denominator

def calculate_slope(numerator, denominator):
    """ this finally finds the slope by dividing numerator
        and denominator"""
    slope = numerator/denominator
    return slope

def calculate_y_int(x_mean, y_mean, slope):
    """ rearrange the y = mx + b formula to solve for b"""
    b = y_mean - (slope * x_mean)

    return b
# ------------------------------------------------------------------------------#

# THESE FUNCTIONS ARE TO GET X AND Y VALUES FOR LINEAR REGRESSION LINE

def choose_x_val_for_line(x):
    """ use a min and max value as points """
    x_min = min(x) 
    x_max = max(x)
    x_values = [x_min, x_max]
    return x_values

# once have x values need to calculate predicted y values using equation
# for x_min, need y_min corresponding value and vice versa for max

def get_y_val_for_line(x_min, x_max, slope, b):
    """ once have x_values, need to calculate predicted y values using equation
        to get y min and y max"""
    y_min = (x_min * slope) + b
    y_max = (x_max * slope) + b
    y_values = [y_min, y_max]
    return y_values

# THIS FUNCTION WILL CALCULATE THE R CORRELATION USING THE ABOVE CALCULATIONS

def calculate_R_correlation(x_diff, x_diff2, y_diff, y_diff2):
    """ stores numerator and denominator information and divides
      to find correlation"""
    numerator = calculate_numerator(x_diff, y_diff)
    denominator = calculate_r_denominator(x_diff2, y_diff2)
    r_correlation = numerator/denominator

    return r_correlation

# ------------------------------------------------------------------------------#

# THE GENERAL RE-USABLE UTILITY CODE FUNCTIONS ARE USED IN AutoData JUPYTER NOTEBOOK 

def load_dataset(filename):
    """this functions create an instance of MyPyTable and loads a table"""
    table = MyPyTable()
    table.load_from_file(filename)
    return table

def clean_duplicates(table, cols, output_file):
    """this function checks for duplicates, if found, it drops them
        and saves changes to output file"""
    duplicates = table.find_duplicates(cols)
    if duplicates:
        table.drop_rows(duplicates)
        table.save_to_file(output_file)
    return duplicates

def display_dataset_info(table, name):
    """this function gets the shape (number of instances) in 
        a table and prints out the number of rows and columns"""
    shape = table.get_shape()
    print(f"This is the number of instances in the {name} dataset: ", shape)

def check_cleaned_duplicates(table, cols):
    """ this just checks for duplicates"""
    return table.find_duplicates(cols)


def DOE_rating_assign(mpg):
    """this function is for the Discretization and assigns mpg certain DOE ratings"""
    mpg = round(mpg)  # to avoid having certain values that don't fit into these categories (either round up or down)
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
    
# this function came from Dr.Sprint!
def compute_bin_frequencies(values, cutoffs):
    freqs = [0 for _ in range(len(cutoffs) - 1)]  # because N + 1 cutoffs

    for value in values:
        if value == max(values):
            freqs[-1] += 1  # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1  # Increment frequency count for the corresponding bin
    return freqs

def compute_equal_width_cutoffs(values, num_bins):
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins
    cutoffs = []
    
    current_value = min(values)
    while current_value < max(values):
        cutoffs.append(current_value)
        current_value += bin_width
    
    cutoffs.append(max(values))
    
    return cutoffs

# this function came from Dr.Sprint! 
def group_by(table, header, group_by_col_name):
    """ this function that groups rows of data by a specific column"""
    group_by_col_index = header.index(group_by_col_name) # gets index of specified column 
    
    # gets rows from that column 
    col = []
    for row in table:
        col.append(row[group_by_col_index])
 
    unique_col_values = sorted(set(col)) # result is a sorted list of unique values from the original list

    group_subtables = []
    for _ in unique_col_values:
        group_subtables.append([])  # adds an empty list for each unique value

    for row in table:
        group_by_val = row[group_by_col_index]
        subtable_index = unique_col_values.index(group_by_val)
        group_subtables[subtable_index].append(row)

    return unique_col_values, group_subtables

# function that will extract the mpg values from the grouped data for each year
def extract_mpg_from_groups(group_subtables, header):
    """this function takes two parameters: grouped subtables: we found in previous function
        and header of the data """
    mpg_index = header.index('mpg') # gets index
    mpg_values_per_year = []
    
    for group in group_subtables:
        mpg_values = [] # for each group, an empty list is initialized to store the mpg values for that group
        for row in group: #iterates over each row in the current group
                mpg = float(row[mpg_index])
                mpg_values.append(mpg)
        mpg_values_per_year.append(mpg_values)
    
    return mpg_values_per_year