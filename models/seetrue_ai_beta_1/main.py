from typing import Any, Dict, List
from fastapi import FastAPI
from pydantic import BaseModel
import ydf
import math

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
  euclidean_distance: float = math.nan


class Output(BaseModel):
  predictions: List[float]
  label_classes: List[str]



@app.post("/predict")
async def predict(example: Example) -> Output:
  # Wrap the example features into a batch i.e., a list. If multiple examples
  # are available at the same time, it is more efficient to group them and run a
  # single prediction with "predict_batch".
  example_batch: Dict[str, List[Any]] = {
      k: [v] for k, v in example.dict().items()
  }

  prediction_batch = model.predict(example_batch).tolist()

  return Output(
    predictions=prediction_batch[0],
    label_classes=label_classes,

  )


@app.post("/predict_batch")
async def predict_batch(example_batch):
  return model.predict(example_batch).tolist()

