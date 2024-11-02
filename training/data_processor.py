import os
import pandas as pd
import logging
import sys

# Set up logging to both file and console without overriding flush
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# File handler for logging to a file
file_handler = logging.FileHandler('dataset_update.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Stream handler for logging to console (and Jupyter notebook)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

# Folders containing your dataset files
data_folder = '../full_dataset'
output_folder = '../full_dataset_labelled'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Mapping of action names to numeric codes
"""
Since we are using TensorFlow in data training and Keras metrics expect integers. 
The our activity label (result) should not be stored as a string (i.e., walking, reading, playing), 
so let's convert it into an integer.
"""
action_map = {
    'walking': 1,
    'reading': 2,
    'playing': 3
}

# Iterate through each file in the folder
file_count = 0
for filename in os.listdir(data_folder):
    if filename.endswith('.csv') and filename != 'config_data.csv':
        try:
            index_number, action = filename.split('_')
            action = action.replace('.csv', '')

            if action in action_map:
                file_path = os.path.join(data_folder, filename)
                data = pd.read_csv(file_path)

                # Add a 'result' column with the mapped value
                data['result'] = action_map[action]

                # Save the modified data to the new directory
                output_file_path = os.path.join(output_folder, filename)
                data.to_csv(output_file_path, index=False)

                # Log progress and file processed
                file_count += 1
                if file_count % 50 == 0:
                    logger.info(f"Processed {file_count} files so far.")

                logger.info(f"Processed file {filename} - Action: {action}")
                sys.stdout.flush()  # Ensure immediate output in Jupyter

            else:
                logger.warning(f"Skipped file {filename} - Action not recognized")

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            sys.stdout.flush()  # Ensure immediate output in Jupyter

logger.info(f"Completed processing {file_count} files.")
sys.stdout.flush()  # Ensure final output is printed