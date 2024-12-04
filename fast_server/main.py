import traceback
from typing import List

import uvicorn
import ydf
from fastapi import FastAPI, status, Response
from pydantic import BaseModel
from starlette.responses import JSONResponse

from utils import logger, preprocess_data

app = FastAPI()

model = ydf.load_model("model")
label_classes = model.label_classes()
logger.info(f"Current label_classes: {label_classes}")

# Label mapping
label_mapping = {
    "1": "walking",
    "3": "playing",
    "2": "reading"
}


class DataBatches(BaseModel):
    timestamp: List[float] = []
    gazepoint_x: List[float] = []
    gazepoint_y: List[float] = []
    pupil_area_right_sq_mm: List[float] = []
    pupil_area_left_sq_mm: List[float] = []
    eye_event: List[str] = []


class Output(BaseModel):
    walking: float
    playing: float
    reading: float
    process_data: int  # Amount of processed data


@app.get('/hello', status_code=status.HTTP_200_OK)
def hello_world(response: Response):
    return {'Welcome to SeeTrue AI!': "data"}


@app.post("/predict")
async def predict(payload: DataBatches):
    try:
        # Preprocess the payload into individual records
        data = payload.model_dump()
        processed_data = preprocess_data(data)
        process_data = len(processed_data)
        # Aggregate processed data into a single batch input
        batch_input = {
            key: [record[key] for record in processed_data if key != "prev_euclidean_distance"]
            for key in processed_data[0] if key != "prev_euclidean_distance"
        }
        prediction_batch = model.predict(batch_input).tolist()

        # Calculate means for each label class
        means = {label_mapping[label]: 0.0 for label in label_mapping.keys()}
        counts = {label_mapping[label]: 0 for label in label_mapping.keys()}

        for prediction in prediction_batch:
            for i, value in enumerate(prediction):
                class_label = label_classes[i]
                mapped_label = label_mapping[class_label]
                means[mapped_label] += value
                counts[mapped_label] += 1

        # Calculate averages of each prediction
        for label in means:
            if counts[label] > 0:
                means[label] /= counts[label]

        # Construct response
        response = Output(
            walking=means["walking"],
            playing=means["playing"],
            reading=means["reading"],
            process_data=process_data
        )
        return response
    except Exception as e:
        trace_back_msg = traceback.format_exc()
        logger.error(f"{str(e)} \n {trace_back_msg}")
        return JSONResponse(content={"Error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080, workers=1, access_log=True)
