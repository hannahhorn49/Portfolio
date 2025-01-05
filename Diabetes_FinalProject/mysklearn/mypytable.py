##############################################
# Programmer: Hannah Horn, Eva Ulrichsen
# Class: CPSC 322-01 Fall 2024
# Programming Assignment #final project
# 12/9/24
# I did not attempt the bonus
# Description: This program contains MyPyTable methods
#########################

import copy
import csv
import random
from tabulate import tabulate

# combined methods from both Hannah and Eva's MyPyTables for simplicity

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

    def save_to_file_condition(self, filename, col_name, condition=None):
        """Save column names and data to a CSV file. If condition is provided,
        follow instruction.

        Args:
            filename(str): relative path for the CSV file to save the contents to.
            col_name(str): header that condition belongs to
            condition(int): value that must be met for outfile

        Notes:
            Use the csv module.
        """

        # open file in write mode
        with open(filename, "w", encoding="utf-8", newline="") as output_file:
            csv_writer = csv.writer(output_file)

            # write headers to the CSV file
            csv_writer.writerow(self.column_names)

            # find the index of col_name
            if col_name not in self.column_names:
                raise ValueError(f"Column '{col_name}' not found in column names.")

            col_index = self.column_names.index(col_name)

            # write rows that meet condition
            for row in self.data:
                if condition is None or row[col_index] == condition:
                    csv_writer.writerow(row)

    def get_instances(self):
        """Computes the dimension of the table (N).

        Returns:
            int: number of rows in the table (N)
        """
        return len(self.data)

    def fancy_get_column(self, col_identifier):
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
        col_index = self.column_names.index(col_identifier)
        column = []
        for row in self.data:
            column.append([row[col_index]])

        return column

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
        unique_rows = []
        duplicates = []
        self.convert_to_numeric()
        for i, row in enumerate(self.data):
            key = []
            for key_column in key_column_names:
                key.append(row[self.column_names.index(key_column)])
            if key in unique_rows:
                duplicates.append(i)
            else:
                unique_rows.append(key)
        return duplicates

    def show_duplicates(self, key_column_names):
        """Returns a list of duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of rows: list of rows of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        unique_rows = []
        duplicates = []
        self.convert_to_numeric()
        for i, row in enumerate(self.data):
            key = []
            for key_column in key_column_names:
                key.append(row[self.column_names.index(key_column)])
            if key in unique_rows:
                duplicates.append(key)
            else:
                unique_rows.append(key)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_data = []
        for row in self.data:
            contains_missing_value = False
            for value in row:
                if value == 'NA':
                    contains_missing_value = True
            if contains_missing_value is False:
                new_data.append(row)
        self.data = new_data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)

        valid_values = [float(row[col_index]) for row in self.data if row[col_index] != "NA"]

        average_column_value = sum(valid_values) / len(valid_values)

        for row in self.data:
            if row[col_index] == 'NA':
                row[col_index] = average_column_value

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
        table = []
        for col_name in col_names:
            col_index = self.column_names.index(col_name)
            valid_values = [float(row[col_index]) for row in self.data if row[col_index] != "NA"]
            if not valid_values:
                pass
            else:
                col_min = min(valid_values)
                col_max = max(valid_values)
                col_mid = (col_min + col_max) / 2
                col_avg = sum(valid_values) / len(valid_values)

                if len(valid_values) % 2 == 0:
                    median_a = sorted(valid_values)[len(valid_values) // 2]
                    median_b = sorted(valid_values)[len(valid_values) // 2 - 1]
                    col_median = (median_a + median_b) / 2
                else:
                    col_median = sorted(valid_values)[len(valid_values) // 2]

                headers = ["attribute", "min", "max", "mid", "avg", "median"]
                temp_table = [col_name, col_min, col_max, col_mid, col_avg, col_median]
                table.append(temp_table)

        return MyPyTable(headers, table)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        self_key_index = [self.column_names.index(col_name) for col_name in key_column_names]
        other_key_index = [other_table.column_names.index(col_name) for col_name in key_column_names]

        joined_data = []
        joined_column_names = self.column_names + [column for column in other_table.column_names if column not in key_column_names]

        for row1 in self.data:
            self_key_values = [row1[i] for i in self_key_index]

            for row2 in other_table.data:
                other_key_values = [row2[i] for i in other_key_index]

                if self_key_values == other_key_values:
                    joined_row = row1 + [row2[i] for i in range(len(row2)) if i not in other_key_index]
                    joined_data.append(joined_row)

        return MyPyTable(column_names=joined_column_names, data=joined_data)

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
        self_key_index = [self.column_names.index(col_name) for col_name in key_column_names]
        other_key_index = [other_table.column_names.index(col_name) for col_name in key_column_names]

        joined_column_names = self.column_names + [col for col in other_table.column_names if col not in key_column_names]

        joined_data = []

        for row1 in self.data:
            self_key_values = [row1[i] for i in self_key_index]
            match_value = False

            for row2 in other_table.data:
                other_key_values = [row2[i] for i in other_key_index]

                if self_key_values == other_key_values:
                    joined_row_dict = {col: "NA" for col in joined_column_names}

                    for i, col_name in enumerate(self.column_names):
                        joined_row_dict[col_name] = row1[i]

                    for i, col_name in enumerate(other_table.column_names):
                        if col_name not in key_column_names:
                            joined_row_dict[col_name] = row2[i]

                    joined_row = [joined_row_dict[col] for col in joined_column_names]
                    joined_data.append(joined_row)
                    match_value = True

            if not match_value:
                joined_row_dict = {col: "NA" for col in joined_column_names}

                for i, col_name in enumerate(self.column_names):
                    joined_row_dict[col_name] = row1[i]

                joined_row = [joined_row_dict[col] for col in joined_column_names]
                joined_data.append(joined_row)

        for row2 in other_table.data:
            other_key_values = [row2[i] for i in other_key_index]

            match_value = False

            for row3 in joined_data:
                join_key_values = [row3[joined_column_names.index(col)] for col in key_column_names]

                if other_key_values == join_key_values:
                    match_value = True

            if not match_value:
                joined_row_dict = {col: "NA" for col in joined_column_names}

                for i, col_name in enumerate(other_table.column_names):
                    joined_row_dict[col_name] = row2[i]

                joined_row = [joined_row_dict[col] for col in joined_column_names]
                joined_data.append(joined_row)

        return MyPyTable(joined_column_names, joined_data)

    def random_subsample_classes(self, filename, col_name, sample_size=1000, random_state=None):
        """ Randomly subsamples a specified number of instances for each class in a dataset
        and writes the resulting subset to a new file.

        Args:
            data (list): Dataset from mypytable.
            output_file (str): Path to the output CSV file.
            class_column (str): Column name representing the class label.
            sample_size (int): Number of instances to sample per class.
            random_state (int): Seed for reproducibility.
        """
        if random_state is not None:
            random.seed(random_state)

        # Find the index of the class label column
        col_index = self.column_names.index(col_name)

        # Group rows by class (manually)
        class_rows = []
        unique_classes = set(row[col_index] for row in self.data)

        for class_label in unique_classes:
            class_group = [row for row in self.data if row[col_index] == class_label]
            if len(class_group) < sample_size:
                raise ValueError(f"Class '{class_label}' has fewer than {sample_size} rows.")
            class_rows.append((class_label, class_group))

        # Subsample rows for each class
        sampled_rows = []
        for class_label, rows in class_rows:
            sampled_rows.extend(random.sample(rows, sample_size))

        # Write to output file
        with open(filename, "w", encoding="utf-8", newline="") as output_file:
            csv_data = csv.writer(output_file)

            # Write headers first
            csv_data.writerow(self.column_names)

            # Write sampled rows
            csv_data.writerows(sampled_rows)
