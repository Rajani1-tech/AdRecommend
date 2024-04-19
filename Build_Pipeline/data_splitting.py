import pandas as pd
from sklearn.model_selection import train_test_split
from Build_Pipeline.data_preprocessing import DataProcessing

class DataSplitter:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path

    def split_data(self, test_size=0.2, random_state=None):
        """
        Splits the dataset into training and testing sets.

        Parameters:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before splitting.

        Returns:
            train_set (DataFrame): The training set.
            test_set (DataFrame): The testing set.
        """
        # Process the CSV file to get a DataFrame
        data_processor = DataProcessing(self.csv_file_path)
        data = data_processor.process_data()

        # Select features (X) and target variable (y)
        X = data[['visitorEmail', 'ad_id']]  # Features
        y = data['clickedOrNot']  # Target variable

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Concatenate features and target variable to create train_set and test_set
        train_set = pd.concat([X_train, y_train], axis=1)
        test_set = pd.concat([X_test, y_test], axis=1)

        print("Data split successful.")  # Print statement to confirm data split
        
        return train_set, test_set


# Path to the CSV file containing the dataset
csv_file_path = "advertisement_final.csv"

# Create an instance of DataSplitter
data_splitter = DataSplitter(csv_file_path)

# Split the data using the split_data method
train_set, test_set = data_splitter.split_data(test_size=0.2, random_state=42)


