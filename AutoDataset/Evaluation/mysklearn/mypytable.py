##############################################
# Programmer: Hannah Horn
# Class: CPSC 322-01, Fall 2024
# Programming Assignment #5
# 10/28/24
# I did not attempt the bonus
#
# Description: This program provides functions that can be reused to work
#               with datasets.
##############################################


"""Module providing a way to pretty print, read files"""
import copy
import csv
from tabulate import tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        if len(self.data) == 0:
            print("The data table is empty.")
            return 0,0
        else:
            number_of_rows = len(self.data)
            number_of_columns = len(self.data[0])
            return number_of_rows, number_of_columns

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """

        #gets column names from self.column names
        column_names = self.column_names

        #first case: check if the col_identifier is a string and handle appropriately
        if isinstance(col_identifier, str):
            if col_identifier in column_names:
                column_index = column_names.index(col_identifier)  # Use index() here
            else:
                raise ValueError("This is an Invalid Column Name")
        #second case: check if the col_identifier is an int for the column index
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(column_names):
                raise ValueError("This is an Invalid Column Index")
            column_index = col_identifier #if valid, store column_index directly from col_identifier

        #third case: the col_identifier isn't a string or int
        else:
            raise ValueError("The Column Identifer must be a column name or index")

        #now need to extract list of values from that column
        column_values = []
        for row in self.data:  #will iterate over entire data
            value = row[column_index]

            if include_missing_values: #checks to see if include_missing_values parameter is true
                column_values.append(value)
            elif include_missing_values is False:
                self.remove_rows_with_missing_values()
                column_values.append(value)
        return column_values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data: #goes through each row in data
            index = 0 #starts index counter at 0
            for value in row: #goes through each value in the row
                try:
                    row[index] = float(value)
                except ValueError:
                    #if value can't be converted, pass means original value is unchanged
                    pass
                index = index + 1

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        cleaned_data = [] # have an empty cleaned data set
        index = 0

        for row in self.data:  # loop through rows of table
            if index not in row_indexes_to_drop:
                cleaned_data.append(row)
            index = index +1
        self.data = cleaned_data

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        #first initialize an empty list
        table = []

        #next open the file
        input_file = open(filename, "r", encoding = "utf-8")
        #use csv reader object
        csv_data = csv.reader(input_file)

        #assign first row of file to be header
        self.column_names = next(csv_data)

        for row in csv_data:
            table.append(row)

        #now need to assign table to self.data
        self.data = table
        self.convert_to_numeric() #calls this instance method

        #close the file
        input_file.close()

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        #first open file in write mode
        output_file = open(filename, "w", encoding="utf-8")
         #reads file line by line using csv writer object
        csv_data = csv.writer(output_file)

        #next need to write headers first then data
        csv_data.writerow(self.column_names)
        csv_data.writerows(self.data)

        #close the file
        output_file.close()

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """

        #first need to initilize an empty dictionary that stores the info from the rows
        #the key is the tuple of values and the value is the index of the first occurence
        key_row_information = {}

        #then also initialize an empty list for the return of indexes of duplicate rows
        duplicate_row_indexes = []

        #loop to match the index to the key_column_names
        key_column_indexes = []
        for col_name in key_column_names:
            key_column_indexes.append(self.column_names.index(col_name))

        index = 0
        #loop to get values from key columns and store in dictionary
        for row in self.data:
            #can create a tuple key from this key column value in this row
            key = ()
            for i in key_column_indexes:
                key = key + (row[i],)

            #now check to see if this row information has been seen before
            if key in key_row_information:
                duplicate_row_indexes.append(index)
            else:
                #since not already in dictionary, store row index in dictionary
                key_row_information[key] = index
            index = index + 1
        #finally return the list of duplicate indexes
        return duplicate_row_indexes


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        cleaned_data = []
        for row in self.data:  # loop through rows of tabke
            no_na_values = True  # assume there is no na values
            for value in row:  # loops through values in the row
                if value == "NA":
                    no_na_values = False  #change the flag
                    break
            if no_na_values:
                cleaned_data.append(row)  # appends the row if no NA was found

        self.data = cleaned_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column_names = self.column_names
        column_index = column_names.index(col_name) #this will store index of col parameter

        #need to get all non-missing values and use that to calculate the mean
        valid_values = []
        for row in self.data:
            value = row[column_index]
            if value != "NA": #only includes non-missing values
                self.convert_to_numeric()
                valid_values.append(value) #call convert to numeric function
        #need to calculate the mean manually
        sum_of_column = sum(valid_values)
        length_of_column = len(valid_values)
        column_mean = sum_of_column/length_of_column

        #finally need to replace missing values with the calculated mean
        for row in self.data:
            value = row[column_index]
            if value == "NA":
                row[column_index] = column_mean
        return self.data
    
    def calculate_median(self, values):
        """
            Calculates the median value of a list to use in the compute_summary_statistics function

            Args: values(list of integer values)

            Returns: the median value of list
        
        """
        if len(values) == 0:
            return None
        #1. need to sort the data from smallest to largest
        sorted_values = sorted(values)
        length = len(sorted_values)

        #2. median calculated differently if length is even or odd
        #odd case
        if length % 2 == 1:
            median = sorted_values[length // 2]
        #even case
        else:
            lower_middle_index = sorted_values[(length //2) -1]
            upper_middle_index = sorted_values[length // 2]
            median = (lower_middle_index + upper_middle_index) / 2
        return median

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """

        #How to solve
        #1. loop through the specified columns and store indexes
        calculated_values = {}

        for col_name in col_names:
            column_index = self.column_names.index(col_name)

            valid_values = []
            self.convert_to_numeric() #call function to ensure all values are numeric

         #2. loop through rows in data, ignore missing values, and add as values to the key column name
            for row in self.data:
                value = row[column_index]
                if value != "NA": #ignore missing values
                    valid_values.append(value) #call convert to numeric function
            calculated_values[col_name] = valid_values
        #3. calculate summary statistics for each column name in a list
        summary_statistics = []

        for col_name, values in calculated_values.items():
            if len(values) > 0: #makes sure list of values isn't empty
                min_value = min(values)
                max_value = max(values)
                mid_value = (min_value + max_value) / 2
                mean_value = sum(values) / len(values)
                median_value = self.calculate_median(values)

                # store values in list format (what test expects)
                summary_statistics.append([col_name, min_value, max_value, mid_value, mean_value, median_value])

        #4. create an instance of MyPyTable that stores the summary stats computed
        summary_statistics_table = MyPyTable(
            column_names = ("attribute" , "min", "max", "mid", "avg", "median"),
            data = summary_statistics
        )

        return summary_statistics_table

    def filter_table_by_key_column_names(self, key_column_names):
        """This is a helper function for perform inner join 
        and will filter a table by the key column names

        Args: 
            table: a table 
            key_column_names(list of str): column names to use as row keys.

        Returns:
            a filtered table 
        """
        key_indexes = []
        for col in key_column_names:
            if col in self.column_names:
                key_indexes.append(self.column_names.index(col))
            else:
                raise ValueError("Column is not found in table")
        #DEBUG: check to make sure key_indexes isn't empty
        if not key_indexes:
            raise ValueError("No valid indexes found for filtering")

        filtered_data = []
        for row in self.data:
            filtered_row = []
            for index in key_indexes:
                filtered_row.append(row[index])
            filtered_data.append(filtered_row)
        return filtered_data

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        #call the filter function on both tables
        self_filtered = self.filter_table_by_key_column_names(key_column_names)
        other_filtered = other_table.filter_table_by_key_column_names(key_column_names)

        # initialize a joined data list
        joined_data = []

        #joining data rows with appropriate column
        for i, self_row in enumerate(self_filtered):
            for j, other_row in enumerate(other_filtered):
                if self_row == other_row:
                    #create a new row with all the columns from self and other_table
                    combined_row = []
                    for value in self.data[i]: #gets the full row from self
                        combined_row.append(value)
                    for l, value in enumerate(other_table.data[j]):
                        if other_table.column_names[l] not in self.column_names:
                            combined_row.append(value)
                    joined_data.append(combined_row)
        combined_column_names = list(set(self.column_names + other_table.column_names))
        return MyPyTable(combined_column_names, joined_data)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        # start with same process as inner join and filter for only key_column_names

        self_filtered = self.filter_table_by_key_column_names(key_column_names)
        other_filtered = other_table.filter_table_by_key_column_names(key_column_names)
        print("key column names in first table: " ,self_filtered)
        print("key column names in second table: " , other_filtered)

        # prep for MyPyTable and create a combined column names list (only unique) and joined rows list
        # combined column names should be Product, Price, Quantity
        combined_column_names = []
        for col in self.column_names:
            combined_column_names.append(col)
        for col in other_table.column_names:
            if col not in combined_column_names:
                combined_column_names.append(col)
        print("this is self.data column names: " , self.column_names)
        print("this is other table column names: ", other_table.column_names)

        print("this is the combined columns for both tables: " , combined_column_names)
        joined_data = []
        matched_self_rows = []

        print(" ")
        #joining data rows with appropriate column
        for i, self_row in enumerate(self_filtered):
            for j, other_row in enumerate(other_filtered):
                if self_row == other_row:
                    matched_self_rows.append(self_row)
                    #create a new row with all the columns from self and other_table
                    combined_row = []
                    for value in self.data[i]: #gets the full row from self
                        combined_row.append(value)
                    for l, value in enumerate(other_table.data[j]):
                        if other_table.column_names[l] not in self.column_names:
                            combined_row.append(value)
                    print("this is the combined row after first inner join: ", combined_row)
                    joined_data.append(combined_row)
        print(" ")
        #  need to identify what indexes in self didn't match in other_table
        print("this is self filtered: ", self_filtered)
        print("this is matched self rows: ", matched_self_rows)
        for i, self_row in enumerate(self_filtered):
            if self_row not in matched_self_rows:
                combined_row = []
                print("these are the remaining index:" , i)
                for value in self.data[i]:
                    combined_row.append(value)
                print("This is the value at first index of combined table: ", combined_row)
                for column in other_table.column_names:
                    if column not in self.column_names:
                        combined_row.append("NA")
                        print("this is what combined row should look like after first NA check: ", combined_row)
                joined_data.append(combined_row)

        print(" ")
        # need to identify what indexes in other_table didn't match in self
        print("this is other filtered: ", other_filtered)
        print("this is matched self rows: ", matched_self_rows)
        for j, other_row in enumerate(other_filtered):
            print("this is what j is: ", j)
            print("this is other_row:", other_row)
            if other_row not in matched_self_rows:
                combined_row = []
                print("these are the remaining index:" , j)
                for col in combined_column_names:
                    if col in other_table.column_names:
                    # If the column is in other_table_row, add its value
                        combined_row.append(other_table.data[j][other_table.column_names.index(col)])
                        print("this is what combined row should look like after second NA check:" , combined_row)
                    else:
                    # Otherwise, add 'NA' for missing columns in other_table
                        combined_row.append('NA')
        
                print("these are the combined rows after BOTH NA: ", combined_row)
                joined_data.append(combined_row)
                print("this is the FINAL joined data:", joined_data)

        print("this should be the joined_data after appending second wave of NAs:" ,joined_data)
        return MyPyTable(combined_column_names, joined_data)
