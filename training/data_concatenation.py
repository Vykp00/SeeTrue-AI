"""
Combine Data for Each Activity: Each activity (walking, reading, and playing)
has a separate DataFrame to accumulate rows from all corresponding files.
Avoid Duplicate Headers: By concatenating DataFrames without resetting headers,
we ensure only one header row appears in each output file.
Logging: Logs each file addition and the final save action to track progress.
"""
import os
import pandas as pd
import logging
import sys
from data_path import LABELLED_DATA_PATH, COMBINED_FILE_PATH
# Set up logging to both file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# File handler for logging to a file
file_handler = logging.FileHandler('data_concatenation.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Stream handler for logging to console (and Jupyter notebook)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

# Directories for input and output
input_folder = LABELLED_DATA_PATH
output_folder = COMBINED_FILE_PATH

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# DataFrames to store combined data for each activity
combined_data = {
    'walking': pd.DataFrame(),
    'reading': pd.DataFrame(),
    'playing': pd.DataFrame()
}

# Iterate over files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        try:
            # Identify the activity types from filenames
            _, activity = filename.split('_')
            activity = activity.replace('.csv', '')

            # Check if the activity is one of the expected types
            if activity in combined_data:
                file_path = os.path.join(input_folder, filename)
                data = pd.read_csv(file_path)

                # Concatenate daa while avoiding duplicate headers
                combined_data[activity] = pd.concat([combined_data[activity], data], ignore_index=True)

                logger.info(f'Added data from {filename} to {activity}.csv')
            else:
                logger.warning(f'Skipped file {filename} - Unexpected activity type')

        except Exception as e:
            logger.error(f'Error processing file {filename}: {e}')
            sys.stdout.flush()

# Save the combined data to separate CSV files
for activity, df in combined_data.items():
    output_file_path = os.path.join(output_folder, f'{activity}.csv')
    df.to_csv(output_file_path, index=False)
    logger.info(f'Saved combined data to {output_file_path}')
    sys.stdout.flush()

logger.info('Completed combining files into walking.csv, reading.csv, and playing.csv')
sys.stdout.flush()