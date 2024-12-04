import logging
import traceback
from typing import Optional

import numpy as np
import re

# Set up logging to both file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
"""
Utils function for feature engineering and preprocessing the data
"""
# Regular expression to extract x, y, and d for distance calculation
pattern = r'FEx(?P<x>[-+]?\d*\.\d+)y(?P<y>[-+]?\d*\.\d+)d(?P<d>[-+]?\d*\.\d+)'


def calculate_euclidean_distance(eye_event: str) -> Optional[float]:
    """Calculate the Euclidean distance from the eye_event field."""
    try:
        match = re.match(pattern, eye_event)
        if match:
            x = float(match.group('x'))
            y = float(match.group('y'))
            d = float(match.group('d'))

            # Calculate the Euclidean Distance
            F = round(np.sqrt(x ** 2 + y ** 2) * d, 4)
            return F
        return None
    except Exception as e:
        trace_back_msg = traceback.format_exc()
        logger.error(f"Error while calculating euclidean distance : {str(e)} \n {trace_back_msg}")
        raise


def remove_na_row(payload: dict) -> dict:
    """
    Remove rows where 'eye_event' is 'NA' from the JSON-like payload.
    Args:
        payload (dict): The JSON-like batch payload.
    Returns:
        dict: The filtered payload with rows where 'eye_event' is 'NA' removed.
    """
    try:
        # Determine the indices of rows to keep (where 'eye_event' is not 'NA')
        valid_indices = [
            i for i, event in enumerate(payload["eye_event"]) if event.strip() != "NA"
        ]

        # Filter the payload by keeping only the valid indices
        filtered_payload = {key: [values[i] for i in valid_indices] for key, values in payload.items()}

        removed_rows = len(payload["eye_event"]) - len(filtered_payload["eye_event"])
        # print(f"Removed {removed_rows} rows with 'NA' in 'eye_event'.")

        return filtered_payload
    except Exception as e:
        trace_back_msg = traceback.format_exc()
        logger.error(f"Error during NA removal : {str(e)} \n {trace_back_msg}")
        # print(f"Error during NA removal: {e}")
        # return payload
        raise


def preprocess_data(payload: dict) -> list:
    """
    Process the incoming batch payload into individual records.
    Args:
        payload (dict): The JSON-like batch payload.
    Returns:
        list: A list of individual processed records.
    """
    try:
        # Step 1: Remove rows with 'NA' in 'eye_event'
        payload = remove_na_row(payload)

        prev_euclidean_distance = None  # Initialize previous distance
        processed_data = []

        # Step 2: Iterate through the filtered records
        for i in range(len(payload['timestamp'])):
            record = {
                "timestamp": payload["timestamp"][i],
                "gazepoint_x": payload["gazepoint_x"][i],
                "gazepoint_y": payload["gazepoint_y"][i],
                "pupil_area_right_sq_mm": payload["pupil_area_right_sq_mm"][i],
                "pupil_area_left_sq_mm": payload["pupil_area_left_sq_mm"][i],
                "eye_event": payload["eye_event"][i],
            }

            # Step 3: Calculate Euclidean Distance
            record["euclidean_distance"] = calculate_euclidean_distance(record["eye_event"])

            # Step 4: Handle prev_euclidean_distance
            if record["euclidean_distance"] is None:
                if record["eye_event"] in ["S", "BB", "BE"]:
                    record["euclidean_distance"] = 0.0
                elif record["eye_event"] == "FB":
                    record["euclidean_distance"] = prev_euclidean_distance or 1.0

            # Set prev_euclidean_distance for the next record
            record["prev_euclidean_distance"] = prev_euclidean_distance
            prev_euclidean_distance = record["euclidean_distance"]

            # Step 5: Append the fully processed record
            processed_data.append(record)

        return processed_data
    except Exception as e:
        trace_back_msg = traceback.format_exc()
        logger.error(f"Error processing data : {str(e)} \n {trace_back_msg}")
        # print(f"Error Error processing data: {e}")
        raise
