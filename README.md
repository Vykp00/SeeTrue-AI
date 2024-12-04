# HOW TO DEPLOY
Model to Fast API and Docker
```
docker build -t seetrue_ai ./fast_server
docker run --rm -p 8080:8080 -d seetrue_ai
```

## Payload

Request Payload to `/predict`
```.json
{
    "timestamp": [1, 2, 3, 4],
    "gazepoint_x": [0.3992, 0.4026, 0.4547, 0.4586],
    "gazepoint_y": [0.5456, 0.7639, 0.9922, 0.7223],
    "pupil_area_right_sq_mm": [0.42, 0.45, 0.43, 0.49],
    "pupil_area_left_sq_mm": [0.42, 0.45, 0.43, 0.49],
    "eye_event": ["FEx1.2y3.4d5.6", "BE", "S", "NA"]
}

```

Expected Response
```.json

{
    "walking": 0.993380814358808,
    "playing": 0.0021739131433592324,
    "reading": 0.004444444771397157,
    "process_data": 138
}

```