import matplotlib.pyplot as plt
from mypytable import MyPyTable
import utils


# FREQUENCY DIAGRAM PLOTTING HERE
def frequency_diagram(filename, column, xlabel, ylabel, title):
    # Create instance of MyPyTable and load data
    table1 = MyPyTable()
    table1.load_from_file(filename)

    # Extract the column of interest
    column_data = table1.get_column(column)

    # Calculate the frequency of unique values
    count = {}
    for value in column_data:
        if value in count:
            count[value] += 1
        else:
            count[value] = 1

    # Create x and y values from the count dictionary
    x = list(count.keys())
    y = list(count.values())

    # Create the frequency diagram
    plt.bar(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# HISTOGRAM PLOTTING HERE
def create_histogram(filename, column, xlabel, ylabel, title):
    # Create instance of MyPyTable and load data
    table1 = MyPyTable()
    table1.load_from_file(filename)

    # Extract the column of interest
    column_data = table1.get_column(column)

    # create histogram
    plt.figure()
    plt.hist(column_data, bins=10, edgecolor="black")  # Use mpg_col directly
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

# SCATTER PLOT HERE
def scatterPlot_w_linear_regression_line_rCorrelation(col_name, x_values, y_values, xlabel, ylabel, title):

    # create instance of MyPyTable and load correct data (no NA because continuous attribute)
    table1 = MyPyTable()
    table1.load_from_file("output_data/auto-data-remove-NA.txt")

    # use get_column functions to get columns of interest
    mpg_col = table1.get_column("mpg")
    column_data = table1.get_column(col_name)  # Use the dynamic column name here

    # call function to calculate mean of columns 
    x_mean, y_mean = utils.calculate_mean(column_data, mpg_col)

    # call function to find difference between value and mean
    x_diff, y_diff = utils.find_difference(column_data, x_mean, mpg_col, y_mean)

    # call function to calculate numerator
    numerator = utils.calculate_numerator(x_diff, y_diff)

    # call function to calculate denominator
    denominator = utils.calculate_xdiff_squared(x_diff)

    # call functions to calculate slope and y-int
    slope = utils.calculate_slope(numerator, denominator)
    y_int = utils.calculate_y_int(x_mean, y_mean, slope)

    # call functions to get x and y values for line
    x_values = utils.choose_x_val_for_line(column_data)
    y_min, y_max = utils.get_y_val_for_line(x_values[0], x_values[1], slope, y_int)
    y_values = [y_min, y_max]

    # Call function to calculate squared differences
    x_diff_squared = utils.calculate_xdiff_squared(x_diff)
    y_diff_squared = utils.calculate_ydiff_squared(y_diff)

    # Call function to calculate R correlation
    r_correlation = utils.calculate_R_correlation(x_diff, x_diff_squared, y_diff, y_diff_squared)

    # Plot the data and regression line
    plt.figure()
    plt.scatter(column_data, mpg_col)
    plt.grid(True)
    plt.plot(x_values, y_values, color='red', label=f'Linear Regression Line (r = {r_correlation:.2f})') 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()  # so it shows the labels
    plt.tight_layout()
    plt.show()
    

# BOX PLOT FOR BONUS HERE
def create_box_plot(mpg_values_per_year, unique_years, xlabel, ylabel, title):
    plt.figure()
    plt.boxplot(mpg_values_per_year, labels=unique_years)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()