from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, status, Response
from pydantic import BaseModel
import ydf
import math

from starlette.responses import JSONResponse

app = FastAPI()

model = ydf.load_model("model")
label_classes = model.label_classes()


class Example(BaseModel):
  timestamp: float = math.nan
  gazepoint_x: float = math.nan
  gazepoint_y: float = math.nan
  pupil_area_right_sq_mm: float = math.nan
  pupil_area_left_sq_mm: float = math.nan
  eye_event: str = ""
  euclidean_distance: Optional[float] = None
  prev_euclidean_distance: Optional[float] = None  # Allow None as a valid value


class Output(BaseModel):
  predictions: List[List[float]]
  label_classes: List[str]
  prev_euclidean_distance: Optional[List[float]] = None # Return prev_euclidean_distance to run interference

@app.get('/hello', status_code=status.HTTP_200_OK)
def hello_world(response: Response):
    return {'Welcome to SeeTrue AI!': "data"}

@app.post("/predict")
async def predict(examples: List[Example]):
  processed_batches = []
  prev_euclidean_distances = None

  for example in examples:
    # Handle prev_euclidean_distance logic based on eye_event and existing value
    if example.eye_event == "FE":
      if example.prev_euclidean_distance is None:
        example.prev_euclidean_distance = example.euclidean_distance
      else:
        example.prev_euclidean_distance = example.euclidean_distance
    elif example.eye_event == "FB":
      if example.prev_euclidean_distance is None:
        example.euclidean_distance = 1.0
        example.prev_euclidean_distance = example.euclidean_distance

    # For other eye_event values, prev_euclidean_distance is not modified

    # Wrap the example features into a batch, excluding prev_euclidean_distance
    processed_batches.append({
      k: v for k, v in example.model_dump().items() if k != "prev_euclidean_distance"
    })
    prev_euclidean_distances = example.prev_euclidean_distance
  # For other eye_event values, prev_euclidean_distance is not modified

  # Transpose the batch for model input
  example_batch: Dict[str, List[Any]] = {
    key: [batch[key] for batch in processed_batches] for key in processed_batches[0]
  }
  print(example_batch)
  print("Previous Euclidean distances:", prev_euclidean_distances)
  # Perform prediction
  prediction_batch = model.predict(example_batch).tolist()

  # Return the prediction along with the updated prev_euclidean_distance
  response = {
    "predictions": prediction_batch,
    "label_classes": label_classes,
    "prev_euclidean_distance": prev_euclidean_distances
  }
  return JSONResponse(content=response, status_code=status.HTTP_200_OK)

#
# @app.post("/predict_batch")
# async def predict_batch(example_batch):
#   return model.predict(example_batch).tolist()
if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8080, workers=1, access_log=True)
