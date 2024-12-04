import traceback
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, status, Response
from pydantic import BaseModel
import ydf
import math
from utils import logger, preprocess_data
from starlette.responses import JSONResponse

app = FastAPI()

model = ydf.load_model("model")
label_classes = model.label_classes()


class DataBatches(BaseModel):
  timestamp: List[float] = []
  gazepoint_x: List[float] = []
  gazepoint_y: List[float] = []
  pupil_area_right_sq_mm: List[float] = []
  pupil_area_left_sq_mm: List[float] = []
  eye_event: List[str] = []


class Output(BaseModel):
  predictions: List[List[float]]
  label_classes: List[str]
  prev_euclidean_distance: Optional[List[float]] = None # Return prev_euclidean_distance to run interference

@app.get('/hello', status_code=status.HTTP_200_OK)
def hello_world(response: Response):
    return {'Welcome to SeeTrue AI!': "data"}

@app.post("/predict")
async def predict(payload: DataBatches):
  try:
    # Preprocess the payload into individual records
    data = payload.model_dump()
    processed_data = preprocess_data(data)

    print("Preprocessd Data: ", processed_data)

    # Aggregate processed data into a single batch input
    batch_input = {
      key: [record[key] for record in processed_data if key != "prev_euclidean_distance"]
      for key in processed_data[0] if key != "prev_euclidean_distance"
    }
    print("Input Data: ", batch_input)
    prediction_batch = model.predict(batch_input).tolist()
    print(prediction_batch)
    # Return the prediction along with the updated prev_euclidean_distance
    response = {
      "predictions": prediction_batch,
      "label_classes": label_classes,
      "batch_input": batch_input,  # Include processed records for verification
    }
    return JSONResponse(content=response, status_code=status.HTTP_200_OK)
  except Exception as e:
    trace_back_msg = traceback.format_exc()
    logger.error(f"{str(e)} \n {trace_back_msg}")
    return JSONResponse(content={"Error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080, workers=1, access_log=True)
