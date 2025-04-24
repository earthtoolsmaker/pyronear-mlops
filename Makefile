.PHONY: check fix mlflow_start mlflow_stop run_yolov8_hyperparameter_search run_yolo_hyperparameter_search yolo_benchmark run_test_suite

check:
	uv run isort --check .
	uv run flake8 .
	uv run mypy .
	uv run black --check .

fix:
	uv run isort .
	uv run black .

mlflow_start:
	uv run mlflow server --backend-store-uri runs/mlflow

mlflow_stop:
	ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9

run_yolo_hyperparameter_search:
	uv run python ./scripts/model/yolo/hyperparameter_search.py \
	  --data ./data/03_model_input/wildfire/full/datasets/data.yaml \
	  --output-dir ./data/04_models/yolo/ \
	  --experiment-name "random_hyperparameter_search" \
	  --filepath-space-yaml ./scripts/model/yolo/spaces/default.yaml \
	  --n 5 \
	  --loglevel "info"

yolo_benchmark:
	uv run python ./scripts/model/yolo/benchmark.py \
	  --input-dir ./data/04_models/yolo/ \
	  --output-dir ./data/06_reporting/yolo/ \
	  --loglevel "info"

run_test_suite:
	uv run pytest
