#!/bin/bash
set -vex

# Pack the model into a Docker endpoint, upload it to Google Cloud, and start a service endpoint.
# Look at the logs for the end-point url.
# Monitor your end-points at: https://pantheon.corp.google.com/run
gcloud run deploy ydf-predict --source . --region us-east1
