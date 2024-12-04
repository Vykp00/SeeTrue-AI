import csv
import json
from collections import defaultdict

# Path to the CSV file
csv_file = 'walking_sample.csv'

# JSON structure initialization
result = defaultdict(list)

# Read and process the CSV file
with open(csv_file, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        result["timestamp"].append(int(row["timestamp"]))
        result["gazepoint_x"].append(float(row["gazepoint_x"]))
        result["gazepoint_y"].append(float(row["gazepoint_y"]))
        result["pupil_area_right_sq_mm"].append(float(row["pupil_area_right_sq_mm"]))
        result["pupil_area_left_sq_mm"].append(float(row["pupil_area_left_sq_mm"]))
        result["eye_event"].append(row["eye_event"].strip())

# Convert the defaultdict to a regular dictionary
result = dict(result)

# Save the JSON to a file (optional) or print it
output_json = 'walking_sample.json'
with open(output_json, 'w') as json_file:
    json.dump(result, json_file, indent=4)

print(json.dumps(result, indent=4))
