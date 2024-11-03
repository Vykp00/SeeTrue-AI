# HOW TO DEPLOY
Model to Fast API and Docker
```
docker build -t seetrue_ai ./fast_server
docker run --rm -p 8080:8080 -d seetrue_ai
```

## Payload

Request Payload
```.json
{
  "timestamp": 5,
  "gazepoint_x": 0.7757,
  "gazepoint_y": 0.1935,
  "pupil_area_right_sq_mm": 0.12,
  "pupil_area_left_sq_mm": 0.2,
  "eye_event": "FE",
  "euclidean_distance": 0.0132,
  "prev_euclidean_distance": null
}
```

Expected Response
```.json
{
    "predictions": [
        0.826665997505188,
        0.1633332520723343,
        0.009999999776482582
    ],
    "label_classes": [
        "1",
        "3",
        "2"
    ],
    "prev_euclidean_distance": 0.0132
}
```
