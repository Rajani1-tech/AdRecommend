import pandas as pd

def filter_data(csv_file, selected_columns):
    """
    Read a CSV file and filter it based on selected columns.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - selected_columns (list): List of column names to select.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing only the selected columns.
    """
    # Read the CSV file
    data = pd.read_csv(csv_file)
    
    # Select the specified columns
    filtered_data = data[selected_columns]
    
    return filtered_data


def preprocess_data(filtered_data):
    """
    Preprocess the filtered data by handling null values and data type conversion.

    Parameters:
    - filtered_data (pd.DataFrame): DataFrame containing the filtered data.

    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    """
    # Check for null values in selected columns
    null_values = filtered_data.isnull().sum()

    # Display columns with null values
    columns_with_null = null_values[null_values > 0].index.tolist()
    print("Columns with null values:", columns_with_null)

    # Display the count of null values in each column
    for column in columns_with_null:
        print(f"Number of null values in {column}: {null_values[column]}")

    # Fill null values in 'stayTime' column with mode
    filtered_data['stayTime'].fillna(filtered_data['stayTime'].mode()[0], inplace=True)
    
    # Drop rows with null values in 'clickedOrNot' column
    filtered_data.dropna(subset=['clickedOrNot'], inplace=True)

    # Check for null values again after preprocessing
    null_values = filtered_data.isnull().sum()
    print("Columns with null values after preprocessing:", null_values[null_values > 0].index.tolist())

    # Convert 'clickedOrNot' values to integers
    filtered_data['clickedOrNot'] = filtered_data['clickedOrNot'].replace('Clicked', 0)
    filtered_data['clickedOrNot'] = filtered_data['clickedOrNot'].astype(int)
    
    # Display unique values in 'clickedOrNot' column
    unique_values = filtered_data['clickedOrNot'].unique()
    print("Unique values in 'clickedOrNot' after conversion:", unique_values)

    return filtered_data

