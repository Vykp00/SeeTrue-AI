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
        print(f"Removed {removed_rows} rows with 'NA' in 'eye_event'.")

        return filtered_payload
    except Exception as e:
        trace_back_msg = traceback.format_exc()
        logger.error(f"Error during NA removal : {str(e)} \n {trace_back_msg}")
        print(f"Error during NA removal: {e}")
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
        print(f"Error Error processing data: {e}")
        raise


data = [{'timestamp': 1, 'gazepoint_x': 0.5502, 'gazepoint_y': 0.315, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': None},
        {'timestamp': 1, 'gazepoint_x': 0.548, 'gazepoint_y': 0.2929, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 1, 'gazepoint_x': 0.5737, 'gazepoint_y': 0.3835, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 1.0},
        {'timestamp': 1, 'gazepoint_x': 0.5721, 'gazepoint_y': 0.3731, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 1, 'gazepoint_x': 0.5992, 'gazepoint_y': 0.3432, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 1, 'gazepoint_x': 0.5974, 'gazepoint_y': 0.3629, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 1, 'gazepoint_x': 0.5993, 'gazepoint_y': 0.3677, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.597y0.363d0.044', 'euclidean_distance': 0.0307,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 1, 'gazepoint_x': 0.5953, 'gazepoint_y': 0.3615, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0307,
         'prev_euclidean_distance': 0.0307},
        {'timestamp': 1, 'gazepoint_x': 0.5931, 'gazepoint_y': 0.3583, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.595y0.361d0.037', 'euclidean_distance': 0.0258,
         'prev_euclidean_distance': 0.0307},
        {'timestamp': 1, 'gazepoint_x': 0.383, 'gazepoint_y': 0.401, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0258},
        {'timestamp': 1, 'gazepoint_x': 0.3814, 'gazepoint_y': 0.3927, 'pupil_area_right_sq_mm': 0.21,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.4073, 'gazepoint_y': 0.4338, 'pupil_area_right_sq_mm': 0.15,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.392y0.432d0.335', 'euclidean_distance': 0.1954,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 2, 'gazepoint_x': 0.0534, 'gazepoint_y': 0.4004, 'pupil_area_right_sq_mm': 0.13,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.1954},
        {'timestamp': 2, 'gazepoint_x': 0.0952, 'gazepoint_y': 0.444, 'pupil_area_right_sq_mm': 0.14,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.176, 'gazepoint_y': 0.4392, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.175, 'gazepoint_y': 0.4428, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.2203, 'gazepoint_y': 0.4366, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.2378, 'gazepoint_y': 0.4287, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.2479, 'gazepoint_y': 0.4505, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.2784, 'gazepoint_y': 0.451, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.241y0.449d0.101', 'euclidean_distance': 0.0515,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 2, 'gazepoint_x': 0.3109, 'gazepoint_y': 0.4657, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0515},
        {'timestamp': 2, 'gazepoint_x': 0.3125, 'gazepoint_y': 0.4471, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.3147, 'gazepoint_y': 0.4672, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.3331, 'gazepoint_y': 0.4404, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.324y0.458d0.159', 'euclidean_distance': 0.0892,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 2, 'gazepoint_x': 0.6394, 'gazepoint_y': 0.446, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0892},
        {'timestamp': 2, 'gazepoint_x': 0.6353, 'gazepoint_y': 0.4387, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.5972, 'gazepoint_y': 0.4511, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 2, 'gazepoint_x': 0.5659, 'gazepoint_y': 0.4571, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.597y0.451d0.034', 'euclidean_distance': 0.0254,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 2, 'gazepoint_x': 0.5656, 'gazepoint_y': 0.4346, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0254,
         'prev_euclidean_distance': 0.0254},
        {'timestamp': 2, 'gazepoint_x': 0.5562, 'gazepoint_y': 0.4212, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.566y0.435d0.053', 'euclidean_distance': 0.0378,
         'prev_euclidean_distance': 0.0254},
        {'timestamp': 3, 'gazepoint_x': 0.5457, 'gazepoint_y': 0.4094, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0378},
        {'timestamp': 3, 'gazepoint_x': 0.7173, 'gazepoint_y': 0.3569, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.6596, 'gazepoint_y': 0.3844, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.6337, 'gazepoint_y': 0.3879, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.7406, 'gazepoint_y': 0.3797, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.7836, 'gazepoint_y': 0.3797, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.7463, 'gazepoint_y': 0.3577, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.7259, 'gazepoint_y': 0.361, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.7058, 'gazepoint_y': 0.3761, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.6902, 'gazepoint_y': 0.3873, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.706y0.376d0.040', 'euclidean_distance': 0.032,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 3, 'gazepoint_x': 0.6718, 'gazepoint_y': 0.365, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.032,
         'prev_euclidean_distance': 0.032},
        {'timestamp': 3, 'gazepoint_x': 0.6228, 'gazepoint_y': 0.3307, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.672y0.365d0.046', 'euclidean_distance': 0.0352,
         'prev_euclidean_distance': 0.032},
        {'timestamp': 3, 'gazepoint_x': 0.5999, 'gazepoint_y': 0.3104, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0352,
         'prev_euclidean_distance': 0.0352},
        {'timestamp': 3, 'gazepoint_x': 0.5936, 'gazepoint_y': 0.3539, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.595y0.321d0.116', 'euclidean_distance': 0.0784,
         'prev_euclidean_distance': 0.0352},
        {'timestamp': 3, 'gazepoint_x': 0.7825, 'gazepoint_y': 0.3709, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0784},
        {'timestamp': 3, 'gazepoint_x': 0.9158, 'gazepoint_y': 0.4116, 'pupil_area_right_sq_mm': 0.15,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 3, 'gazepoint_x': 0.8659, 'gazepoint_y': 0.3618, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.8292, 'gazepoint_y': 0.3349, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.7899, 'gazepoint_y': 0.3444, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.8606, 'gazepoint_y': 0.3284, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.8134, 'gazepoint_y': 0.379, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.7861, 'gazepoint_y': 0.389, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.7823, 'gazepoint_y': 0.3944, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.7695, 'gazepoint_y': 0.362, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.782y0.394d0.055', 'euclidean_distance': 0.0482,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 4, 'gazepoint_x': 0.7458, 'gazepoint_y': 0.3618, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0482},
        {'timestamp': 4, 'gazepoint_x': 0.7299, 'gazepoint_y': 0.3582, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.7021, 'gazepoint_y': 0.3715, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.730y0.358d0.049', 'euclidean_distance': 0.0398,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 4, 'gazepoint_x': 0.6817, 'gazepoint_y': 0.3957, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0398,
         'prev_euclidean_distance': 0.0398},
        {'timestamp': 4, 'gazepoint_x': 0.7722, 'gazepoint_y': 0.4086, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.682y0.396d0.060', 'euclidean_distance': 0.0473,
         'prev_euclidean_distance': 0.0398},
        {'timestamp': 4, 'gazepoint_x': 0.8231, 'gazepoint_y': 0.3477, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0473},
        {'timestamp': 4, 'gazepoint_x': 0.8193, 'gazepoint_y': 0.351, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 4, 'gazepoint_x': 0.8161, 'gazepoint_y': 0.3517, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.819y0.351d0.040', 'euclidean_distance': 0.0356,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 4, 'gazepoint_x': 0.8032, 'gazepoint_y': 0.3541, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0356,
         'prev_euclidean_distance': 0.0356},
        {'timestamp': 5, 'gazepoint_x': 0.6201, 'gazepoint_y': 0.4321, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.783y0.367d0.199', 'euclidean_distance': 0.1721,
         'prev_euclidean_distance': 0.0356},
        {'timestamp': 5, 'gazepoint_x': 0.5257, 'gazepoint_y': 0.392, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.1721},
        {'timestamp': 5, 'gazepoint_x': 0.5149, 'gazepoint_y': 0.4366, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.4807, 'gazepoint_y': 0.4644, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'BE', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.4258, 'gazepoint_y': 0.4371, 'pupil_area_right_sq_mm': 0.24,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.4227, 'gazepoint_y': 0.4271, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.4574, 'gazepoint_y': 0.4267, 'pupil_area_right_sq_mm': 0.21,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.4416, 'gazepoint_y': 0.4222, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.4289, 'gazepoint_y': 0.4373, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.442y0.422d0.039', 'euclidean_distance': 0.0238,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 5, 'gazepoint_x': 0.491, 'gazepoint_y': 0.4248, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0238},
        {'timestamp': 5, 'gazepoint_x': 0.587, 'gazepoint_y': 0.4387, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.5931, 'gazepoint_y': 0.4599, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 5, 'gazepoint_x': 0.6156, 'gazepoint_y': 0.4619, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.593y0.460d0.054', 'euclidean_distance': 0.0405,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 5, 'gazepoint_x': 0.6158, 'gazepoint_y': 0.4431, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0405,
         'prev_euclidean_distance': 0.0405},
        {'timestamp': 5, 'gazepoint_x': 0.6124, 'gazepoint_y': 0.4444, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.613y0.442d0.100', 'euclidean_distance': 0.0756,
         'prev_euclidean_distance': 0.0405},
        {'timestamp': 5, 'gazepoint_x': 0.6733, 'gazepoint_y': 0.4673, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0756},
        {'timestamp': 5, 'gazepoint_x': 0.6809, 'gazepoint_y': 0.501, 'pupil_area_right_sq_mm': 0.21,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7305, 'gazepoint_y': 0.5221, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.660y0.493d0.202', 'euclidean_distance': 0.1664,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 6, 'gazepoint_x': 0.8598, 'gazepoint_y': 0.5237, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.1664},
        {'timestamp': 6, 'gazepoint_x': 0.8293, 'gazepoint_y': 0.5271, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.8066, 'gazepoint_y': 0.5324, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.8012, 'gazepoint_y': 0.5054, 'pupil_area_right_sq_mm': 0.16,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.786, 'gazepoint_y': 0.5081, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7589, 'gazepoint_y': 0.4971, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7822, 'gazepoint_y': 0.4057, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7815, 'gazepoint_y': 0.4088, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7784, 'gazepoint_y': 0.4424, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7684, 'gazepoint_y': 0.4295, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.7625, 'gazepoint_y': 0.4417, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.768y0.430d0.058', 'euclidean_distance': 0.0511,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 6, 'gazepoint_x': 0.7561, 'gazepoint_y': 0.4496, 'pupil_area_right_sq_mm': 0.21,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0511},
        {'timestamp': 6, 'gazepoint_x': 0.5405, 'gazepoint_y': 0.3489, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.5418, 'gazepoint_y': 0.3827, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 6, 'gazepoint_x': 0.5513, 'gazepoint_y': 0.401, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.546y0.384d0.100', 'euclidean_distance': 0.0668,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 7, 'gazepoint_x': 0.5617, 'gazepoint_y': 0.4246, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0668},
        {'timestamp': 7, 'gazepoint_x': 0.5699, 'gazepoint_y': 0.4435, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'BE', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.7529, 'gazepoint_y': 0.4443, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.7447, 'gazepoint_y': 0.4312, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.7613, 'gazepoint_y': 0.4283, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.7612, 'gazepoint_y': 0.4281, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.7521, 'gazepoint_y': 0.378, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.744, 'gazepoint_y': 0.3844, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.7364, 'gazepoint_y': 0.383, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.744y0.384d0.044', 'euclidean_distance': 0.0368,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 7, 'gazepoint_x': 0.7364, 'gazepoint_y': 0.383, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0368},
        {'timestamp': 7, 'gazepoint_x': 0.3049, 'gazepoint_y': 0.5075, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.3024, 'gazepoint_y': 0.5185, 'pupil_area_right_sq_mm': 0.24,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.3468, 'gazepoint_y': 0.518, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.3496, 'gazepoint_y': 0.5134, 'pupil_area_right_sq_mm': 0.24,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 7, 'gazepoint_x': 0.3535, 'gazepoint_y': 0.4877, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 8, 'gazepoint_x': 0.3818, 'gazepoint_y': 0.4773, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.356y0.485d0.114', 'euclidean_distance': 0.0686,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 8, 'gazepoint_x': 0.3873, 'gazepoint_y': 0.4699, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.0686,
         'prev_euclidean_distance': 0.0686},
        {'timestamp': 8, 'gazepoint_x': 0.2538, 'gazepoint_y': 0.5172, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.388y0.464d0.071', 'euclidean_distance': 0.0429,
         'prev_euclidean_distance': 0.0686},
        {'timestamp': 8, 'gazepoint_x': 0.2681, 'gazepoint_y': 0.4996, 'pupil_area_right_sq_mm': 0.21,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0429},
        {'timestamp': 8, 'gazepoint_x': 0.2815, 'gazepoint_y': 0.4966, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 8, 'gazepoint_x': 0.2496, 'gazepoint_y': 0.5379, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.262y0.496d0.220', 'euclidean_distance': 0.1234,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 8, 'gazepoint_x': 0.3494, 'gazepoint_y': 0.5248, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.1234},
        {'timestamp': 8, 'gazepoint_x': 0.3455, 'gazepoint_y': 0.5464, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 8, 'gazepoint_x': 0.3445, 'gazepoint_y': 0.5556, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 8, 'gazepoint_x': 0.403, 'gazepoint_y': 0.5126, 'pupil_area_right_sq_mm': 0.23,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.349y0.554d0.135', 'euclidean_distance': 0.0884,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 8, 'gazepoint_x': 0.5527, 'gazepoint_y': 0.484, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0884},
        {'timestamp': 8, 'gazepoint_x': 0.5548, 'gazepoint_y': 0.4692, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 8, 'gazepoint_x': 0.5544, 'gazepoint_y': 0.4799, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.5561, 'gazepoint_y': 0.4921, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.554y0.484d0.161', 'euclidean_distance': 0.1184,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 9, 'gazepoint_x': 0.5521, 'gazepoint_y': 0.4897, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 0.1184,
         'prev_euclidean_distance': 0.1184},
        {'timestamp': 9, 'gazepoint_x': 0.5291, 'gazepoint_y': 0.4501, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.542y0.472d0.178', 'euclidean_distance': 0.1279,
         'prev_euclidean_distance': 0.1184},
        {'timestamp': 9, 'gazepoint_x': 0.529, 'gazepoint_y': 0.4484, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'BE', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.1279},
        {'timestamp': 9, 'gazepoint_x': 0.4423, 'gazepoint_y': 0.4311, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.4304, 'gazepoint_y': 0.407, 'pupil_area_right_sq_mm': 0.22,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.4103, 'gazepoint_y': 0.4366, 'pupil_area_right_sq_mm': 0.21,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.4116, 'gazepoint_y': 0.4383, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.3907, 'gazepoint_y': 0.4512, 'pupil_area_right_sq_mm': 0.2,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.4612, 'gazepoint_y': 0.5183, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.402y0.427d0.247', 'euclidean_distance': 0.1449,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 9, 'gazepoint_x': 0.5223, 'gazepoint_y': 0.5138, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.1449},
        {'timestamp': 9, 'gazepoint_x': 0.5208, 'gazepoint_y': 0.5114, 'pupil_area_right_sq_mm': 0.18,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FB', 'euclidean_distance': 1.0, 'prev_euclidean_distance': 0.0},
        {'timestamp': 9, 'gazepoint_x': 0.5043, 'gazepoint_y': 0.5087, 'pupil_area_right_sq_mm': 0.19,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'FEx0.521y0.511d0.037', 'euclidean_distance': 0.027,
         'prev_euclidean_distance': 1.0},
        {'timestamp': 9, 'gazepoint_x': 0.537, 'gazepoint_y': 0.4756, 'pupil_area_right_sq_mm': 0.17,
         'pupil_area_left_sq_mm': 0.0, 'eye_event': 'S', 'euclidean_distance': 0.0, 'prev_euclidean_distance': 0.027}]
