import os
import pandas as pd
import logging
import sys
from sklearn.model_selection import train_test_split
from data_path import FEATURE_FILE_PATH, DATA_SPLIT_PATH

# Set up logging to both file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# File handler for logging to a file
file_handler = logging.FileHandler('create_train_test_data.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(file_handler)

# Stream handler for logging to console (and Jupyter notebook)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(console_handler)

# Directory containing the combined files
input_folder = FEATURE_FILE_PATH
output_folder = DATA_SPLIT_PATH
os.makedirs(output_folder, exist_ok=True)

# Load each activity file with Result column as Y indicator
activity_files = {
    'walking': 1,
    'reading': 2,
    'playing': 3
}

# List to store all data for concatenation
all_data = []

# Load each file and append to all_data
for activity, label in activity_files.items():
    file_path = os.path.join(input_folder, f"{activity}.csv")
    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            data['Result'] = label  # Set the Result column for classification target
            all_data.append(data)
            logger.info(f"Loaded data for {activity} with label {label}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            sys.stdout.flush()

# Concatenate all data into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)
logger.info("Combined all data for walking, reading, and playing")

# Split each class separately to ensure balanced classes in train and test sets
train_data = []
test_data = []

for label in activity_files.values():
    class_data = combined_data[combined_data['Result'] == label]
    train, test = train_test_split(class_data, test_size=0.2, random_state=42)
    train_data.append(train)
    test_data.append(test)
    logger.info(f"Split data for label {label} into training and testing")

# Concatenate all classes to form the final train and test datasets
train_data = pd.concat(train_data, ignore_index=True)
test_data = pd.concat(test_data, ignore_index=True)

# Save train and test datasets
train_data.to_csv(os.path.join(output_folder, 'train.csv'), index=False)
test_data.to_csv(os.path.join(output_folder, 'test.csv'), index=False)

logger.info("Saved train and test datasets to 'dataset_split' folder")
sys.stdout.flush()
