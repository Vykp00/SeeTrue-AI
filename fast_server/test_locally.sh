#!/bin/bash
set -vex

# Pack the model into a Docker endpoint
docker build --platform linux/amd64 -t ydf_predict_image .

# Start the docker endpoint locally
CONTAINER_NAME=$(docker run --rm -p 8080:8080 -d ydf_predict_image)

# Wait for the endpoint to load
sleep 5

# Send a prediction request to the endpoint
curl -X 'POST'   'localhost:8080/predict'   -H "accept: application/json"   -H 'Content-Type: application/json'   -d '{}'

# Stop the endpoint
docker stop ${CONTAINER_NAME}
