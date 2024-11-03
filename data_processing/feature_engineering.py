import os
import pandas as pd
import logging
import sys
import numpy as np
import re

from data_path import COMBINED_FILE_PATH, FEATURE_FILE_PATH

# Set up logging to both file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# File handler for logging to a file
file_handler = logging.FileHandler('feature_engineering.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Stream handler for logging to console (and Jupyter notebook)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

def remove_na_row(dataframe_data, file_name):
    try:
        # Drop the row where 'Eye event' column has 'NA' values
        original_count = len(dataframe_data)
        dataframe_data = dataframe_data[dataframe_data[' Eye event '] != ' NA ']
        # Count the remaining rows
        filtered_count = len(dataframe_data)

        # Saved the filtered DataFrame and log the number of row removed
        removed_rows = original_count - filtered_count

        logger.info(f"{removed_rows} rows with 'NA' in 'Eye event' removed from {file_name} .")

        return dataframe_data
    except Exception as e:
        logger.error(f"Error processing file {file_name} : {e}")

# Regular expression to match and extract x, y, and d coordinates
pattern = r' FEx(?P<x>[-+]?\d*\.\d+)y(?P<y>[-+]?\d*\.\d+)d(?P<d>[-+]?\d*\.\d+) '

def euclidean_distance_cal(dataframe_data, file_name):
    try:
        # Process each row to calculate Euclidean Distance where pattern matches
        distances = []
        for i, row in dataframe_data.iterrows():
            eye_event = row[' Eye event ']
            match = re.match(pattern, eye_event)

            if match:
                # Extract x, y, and d as floats
                x = float(match.group('x'))
                y = float(match.group('y'))
                d = float(match.group('d'))

                # Calculate the Euclidean Distance
                F = np.sqrt(x ** 2 + y ** 2) * d
                F = round(F, 4)  # Round to 4 decimal places
                distances.append(F)

                # Update ' Eye event ' to ' FE '
                dataframe_data.at[i, ' Eye event '] = " FE "

            else:
                # No change for rows without matching pattern
                # We will handle this value later
                distances.append(np.nan)
        # Add new column 'Euclidean Distance' before 'Result' column
        dataframe_data.insert(dataframe_data.columns.get_loc('Result'), 'Euclidean Distance', distances)

        logger.info(f"Calculated Euclidean Distances for {file_name} .")

        return dataframe_data
    except Exception as e:
        logger.error(f"Error processing file {file_name} : {e}")


def replace_nan_euclidean_distance(dataframe_data, file_name):
    """
    Now Euclidean Distance is calculated, we still have to handle the NaN values in this rows. For this, we set a rule:
        * If the eye event is `'S`, `BB`, `BE`
        * If the eye event is `FB`, we can save it as the previous Euclidean Distance we calculated,
            if none has been calculated, we set it as `1.0`
    """
    try:
        # Ensure 'Euclidean Distance' column exists
        if 'Euclidean Distance' not in dataframe_data.columns:
            return dataframe_data  # If the column doesn't exist, return the DataFrame as is

        previous_value = None  # Start with no previous value

        for i, row in dataframe_data.iterrows():
            eye_event = row[' Eye event '].strip()  # Stripping any whitespace

            if pd.isna(row['Euclidean Distance']):
                if eye_event in ['S', 'BB', 'BE']:
                    # Set NaN to 0.0 for 'S', 'BB', or 'BE'
                    dataframe_data.at[i, 'Euclidean Distance'] = 0.0
                elif eye_event == 'FB':
                    # Set NaN to the previous non-NaN value or 1.0 if not found
                    if previous_value is not None:
                        dataframe_data.at[i, 'Euclidean Distance'] = previous_value
                    else:
                        dataframe_data.at[i, 'Euclidean Distance'] = 1.0
            else:
                # Update previous_value only for non-NaN entries
                previous_value = row['Euclidean Distance']

        logger.info(f"Replaced all NaN values in Euclidean Distances for {file_name} .")

        return dataframe_data
    except Exception as e:
        logger.error(f"Error processing file {file_name} : {e}")

# Now to call all submodules and process file
def process_files(input_directory, output_directory):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    # List of CSV files to process
    files_to_process = ['walking.csv', 'reading.csv', 'playing.csv']
    for filename in files_to_process:
        try:
            # Read each CSV files
            file_path = os.path.join(input_directory, filename)
            logger.info(f"Processing {filename}...")
            df = pd.read_csv(file_path)

            # Step 1: Remove rows with 'NA' in 'Eye event'
            df = remove_na_row(df, filename)

            # Step 2: Calculate Euclidean Distance
            df = euclidean_distance_cal(df, filename)

            # Step 3: Replace NaN values in Euclidean Distance
            df = replace_nan_euclidean_distance(df, filename)

            # Save the processed DataFrame to a new CSV file
            output_file_path = os.path.join(output_directory, filename)
            df.to_csv(output_file_path, index=False)

            logger.info(f"Saved processed file to {output_file_path}")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error processing file {filename} : {e}")
            sys.stdout.flush()

# Execute when run
if __name__ == '__main__':
    input_dir = COMBINED_FILE_PATH
    output_dir = FEATURE_FILE_PATH
    process_files(input_dir, output_dir)