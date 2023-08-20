#!/bin/bash

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
EXECUTION_DATE=$(date "+%Y-%m-%d")
ROOT_DIR=$PWD
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "Timestamp: $TIMESTAMP"
echo "Execution date: $EXECUTION_DATE"
echo "Root directory: $ROOT_DIR"
echo "Log directory: $LOG_DIR"


# Env variables
export AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export MLFLOW_SERVER_USERNAME=$MLFLOW_SERVER_USERNAME
export MLFLOW_SERVER_PASSWORD=$MLFLOW_SERVER_PASSWORD

# Execution
papermill "$ROOT_DIR/notebooks/house_pricing_model_building_deployed.ipynb" \
"$LOG_DIR/${TIMESTAMP}-house_pricing_model_building_deployed.ipynb"

papermill "$ROOT_DIR/notebooks/house_pricing_analyse.ipynb" \
"$LOG_DIR/${TIMESTAMP}-house_pricing_analyse.ipynb"