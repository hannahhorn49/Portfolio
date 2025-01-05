##############################################
# Programmer: Hannah Horn, Eva Ulrichsen
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #final project
# 12/9/24
# I did not attempt the bonus
# Description: This program contains graphing functions for the dataset.
#########################

from mysklearn.mypytable import MyPyTable
import mysklearn.utils as utils
import numpy as np
import matplotlib.pyplot as plt

def categorical_frequencies_bar(data, headers, column_name):
    """Creates a bar graph based on categorical values

    Args:
        column_name(str or int): string for a column name or int
            for a column index
        data(list of obj): 2D data structure 
        headers(list of str): M column names
    """
    values, counts = utils.get_frequencies(data, headers, column_name)
    # print(values)
    # print(counts)

    plt.bar(values, counts, width=(values[1] - values[0]),
            edgecolor="black", align="edge")

    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title("Total Number by " + column_name)
    plt.show()

def frequencies_scatter(data, headers, column_name):
    """Creates a scatter plot of frequency counts for a categorical column.

    Args:
        data (list of lists): 2D dataset.
        headers (list of str): List of column names.
        column_name (str or int): Column name or index.
    """
    values, counts = utils.get_frequencies(data, headers, column_name)
    # print(values)
    # print(counts)

    plt.scatter(values, counts)

    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.title("Total Number by " + column_name)
    plt.show()

def continuous_relationship_plt(data, header, col_name1, col_name2):
    """Creates a plot that compares two columns and computes a slope and
        r coefficient value

    Args:
        col_name1(str or int): string for a column name or int
            for a column index
        col_name2(str or int): string for a column name or int
            for a column index
        data(list of obj): 2D data structure 
        header(list of str): M column names
    """
    x = utils.get_column(data, header, col_name1)
    y = utils.get_column(data, header, col_name2)

    m, b = utils.compute_slope_intercept(x, y)
    r = utils.calculate_r_value(x, y)

    print(m, b)
    plt.scatter(x, y)
    plt.grid(True)
    plt.plot([min(x), max(x)], [m * min(x) + b, m * max(x) + b], c="orange", lw=3)
    plt.text(20, 49, f"r = {r:.2f}", fontsize=12, color='red', ha='center')
    plt.title(col_name1 + " vs " + col_name2)
    plt.xlabel(col_name2)
    plt.ylabel(col_name1)

def basic_histogram(table, headers, col_name, bin_count):
    """Creates a basic histogram graph with given bin count

    Args:
        col_name(str or int): string for a column name or int
            for a column index
        table(list of obj): 2D data structure 
        headers(list of str): M column names
        bin_count(int): int of bin count
    """
    values = utils.get_column(table, headers, col_name)

    plt.figure()
    plt.hist(values, bins=bin_count, facecolor="pink", edgecolor="black")
    plt.xlabel(col_name)
    plt.ylabel("Frequency")
    plt.title("Distribution of " + col_name + " Values")
    plt.show()

def continuous_box_plt(data, headers, col_name):
    """Creates a box plot comparing a continuous variable by diabetes status.

    Args:
        data (list of lists): 2D dataset.
        headers (list of str): List of column names.
        col_name (str): Continuous column name.
    """
    x = utils.get_column(data, headers, col_name)
    y = utils.get_column(data, headers, "diabetes")

    neg_y = [val for val, status in zip(x, y) if status == 0]
    pos_y = [val for val, status in zip(x, y) if status == 1]

    data = [neg_y, pos_y]
    labels = ['No Diabetes', 'Diabetes']

    plt.figure()
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))

    plt.title(col_name + " by Diabetes Status")
    plt.ylabel(col_name)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

def grouped_bar_chart(data, header, col_name):
    """Creates a grouped bar chart comparing a categorical variable to diabetes status.

    Args:
        data (list of lists): 2D dataset.
        header (list of str): List of column names.
        col_name (str or int): Column name or index for the categorical variable.
    """
    x = utils.get_column(data, header, col_name)
    y = utils.get_column(data, header, "diabetes")

    x = [int(xi) for xi in x]
    y = [int(yi) for yi in y]

    unique_x = sorted(set(x))
    y_counts = {key: [0, 0] for key in unique_x}

    for xi, yi in zip(x, y):
        y_counts[xi][yi] += 1

    x_indices = np.arange(len(unique_x))
    bar_width = 0.4

    y_0_counts = [y_counts[xi][0] for xi in unique_x]
    y_1_counts = [y_counts[xi][1] for xi in unique_x]

    plt.bar(x_indices - bar_width / 2, y_0_counts, width=bar_width, label=col_name + " = 0", color="skyblue")
    plt.bar(x_indices + bar_width / 2, y_1_counts, width=bar_width, label=col_name + " = 1", color="salmon")

    plt.title("Grouped Bar Chart of " + col_name + " versus diabetes")
    plt.xlabel("diabetes")
    plt.ylabel("Count")
    plt.xticks(x_indices, labels=unique_x)
    plt.legend(title=col_name)

    plt.tight_layout()
    plt.show()

def categorical_vs_binary_plt(data, header, categorical_col, binary_col):
    """
    Creates a grouped bar chart comparing a categorical variable to a binary variable.

    Args:
        data (list of lists): The dataset.
        header (list of str): Column headers for the dataset.
        categorical_col (str): Name of the categorical variable (e.g., "smoking history").
        binary_col (str): Name of the binary variable (e.g., "diabetes").
    """
    x = utils.get_column(data, header, categorical_col)
    y = utils.get_column(data, header, binary_col)

    y = [int(yi) for yi in y]

    unique_x = sorted(set(x))
    y_counts = {key: [0, 0] for key in unique_x}

    for xi, yi in zip(x, y):
        y_counts[xi][yi] += 1

    x_indices = np.arange(len(unique_x))
    bar_width = 0.4

    y_0_counts = [y_counts[xi][0] for xi in unique_x]
    y_1_counts = [y_counts[xi][1] for xi in unique_x]

    plt.figure(figsize=(8, 6))
    plt.bar(x_indices - bar_width / 2, y_0_counts, width=bar_width, label=binary_col + " = 0", color="skyblue")
    plt.bar(x_indices + bar_width / 2, y_1_counts, width=bar_width, label=binary_col + " = 1", color="salmon")

    plt.title(f"Grouped Bar Chart of {categorical_col} vs {binary_col}")
    plt.xlabel(categorical_col)
    plt.ylabel("Count")
    plt.xticks(x_indices, labels=unique_x, rotation=45, ha='right')  # Rotate for better readability
    plt.legend(title=binary_col)

    plt.tight_layout()
    plt.show()

def plot_diabetes_status():

    # get the diabetes column
    # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file("output_data/diabetes_minimize.csv")

    diabetes_column = table.get_column("diabetes")

    # count the occurences of 0 and 1 in the diabetes column
    diabetes_counts = [diabetes_column.count(0) , diabetes_column.count(1)]

    labels = ["No Diabetes", "Diabetes"]

    plt.figure()
    plt.pie(diabetes_counts, labels = labels)
    plt.title("Diabetes Distribution")
    plt.show()


def plot_gender_distribution():
    """Plot a pie chart for the gender distribution (Male vs Female)."""
     # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file("output_data/diabetes_minimize.csv")

    gender_column = table.get_column("gender")

    male_count = gender_column.count("Male")
    female_count = gender_column.count("Female")

    labels = ["Male", "Female"]

    plt.figure()
    plt.pie([male_count, female_count], labels = labels)
    plt.title("Gender Distribution")
    plt.show()

# Hannah plot_utils
def plot_diabetes_by_gender():
    """Plot a bar chart for diabetes distribution by gender (show counts for each gender and diabetes status)."""

     # create instance of MyPyTable class and load the data
    table = MyPyTable()
    table.load_from_file("output_data/diabetes_minimize.csv")
    
    # extract 'diabetes' and 'gender' columns
    diabetes_column = table.get_column("diabetes")
    gender_column = table.get_column("gender")

    # count the number of diabetic and non-diabetic individuals by gender
    male_diabetes = 0
    female_diabetes = 0
    male_no_diabetes = 0
    female_no_diabetes = 0

    for i in range(len(gender_column)):
        if gender_column[i] == "Male":
            if diabetes_column[i] == 1:
                male_diabetes += 1
            else:
                male_no_diabetes += 1
        elif gender_column[i] == "Female":
            if diabetes_column[i] == 1:
                female_diabetes += 1
            else:
                female_no_diabetes += 1

    # data for the bar chart
    labels = ['Male', 'Female']
    diabetes_counts = [male_diabetes, female_diabetes]
    no_diabetes_counts = [male_no_diabetes, female_no_diabetes]

    n = len(labels)

    # use numpy to create positions of bars
    x = np.arange(n)
    bar_width = 0.35

    fig, ax = plt.subplots()

    # plot bars for Diabetes and No Diabetes
    ax.bar(x - bar_width/2, diabetes_counts, bar_width, label='Diabetes', color='green')
    ax.bar(x + bar_width/2, no_diabetes_counts, bar_width, label='No Diabetes', color='red')

    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    ax.set_title('Diabetes Distribution by Gender')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    plt.show()
