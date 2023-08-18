#! bin/bash

TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
EXECUTION_DATE=$(date "+%Y-%m-%d")
echo "Timestamp: $TIMESTAMP"
echo "Execution date: $EXECUTION_DATE"

ROOT_DIR=$PWD
LOG_DIR="${ROOT_DIR}/logs"
echo "Root directory: $ROOT_DIR"
echo "Log directory: $LOG_DIR"

mkdir -p "${LOG_DIR}"

# Execution du script

papermill "$ROOT_DIR/notebooks/tuto-mlflow-sklearn.ipynb" \
"$LOG_DIR/${TIMESTAMP}-house-price-training.ipynb"