import pandas as pd
import numpy as np

class DataProcessing:
    """
    This class performs basic preprocessing steps, like columns arrangement and renaming as required in the algorithm.

    Attributes:
        csv_path (str): Path of the CSV file.
    """
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def process_data(self):
        """
        Arranging the received csv file into the format best suited for a specific algorithm.

        Returns:
            Preprocessed dataframe.
        """
        # Read the CSV file into a DataFrame
        data = pd.read_csv(self.csv_path)
        
        # Reorder columns and rename them
        data = data[['visitorEmail', 'ad_id', 'clickedOrNot']]
        
        # Replace 'Clicked' and 'Not Clicked' with 1 and 0, respectively
        data['clickedOrNot'] = data['clickedOrNot'].replace({'Clicked': 1, 'Not Clicked': 0})
        
        # Handle non-finite values (NaN or inf) by replacing them with a default value (0)
        data['clickedOrNot'] = data['clickedOrNot'].replace([np.inf, -np.inf, np.nan], 0)
        
        # Convert the column to integers
        data['clickedOrNot'] = data['clickedOrNot'].astype(int)
        
        data = pd.DataFrame(data)
        print(data.head())
        print("Data processed successfully.")
      
        return data

# Example usage:
csv_file_path = "advertisement_final.csv"



# Create an instance of DataProcessing to preprocess the data
data_processor = DataProcessing(csv_file_path)
processed_data = data_processor.process_data()
