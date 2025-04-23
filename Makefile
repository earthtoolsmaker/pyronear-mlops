.PHONY: check fix mlflow_start mlflow_stop run_yolov8_hyperparameter_search run_yolo_hyperparameter_search yolo_benchmark run_test_suite

check:
	poetry run isort --check .
	poetry run flake8 .
	poetry run mypy .
	poetry run black --check .

fix:
	poetry run isort .
	poetry run black .

mlflow_start:
	poetry run mlflow server --backend-store-uri runs/mlflow

mlflow_stop:
	ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9

run_yolo_hyperparameter_search:
	poetry run python ./scripts/model/yolo/hyperparameter_search.py \
	  --data ./data/03_model_input/wildfire/full/datasets/data.yaml \
	  --output-dir ./data/04_models/yolov12/ \
	  --experiment-name "random_hyperparameter_search" \
	  --model-versions 12 11 \
	  --model-sizes "n" "s" "m" \
	  --batch-sizes 16 \
	  --n 5 \
	  --loglevel "info"

yolo_benchmark:
	poetry run python ./scripts/model/yolo/benchmark.py \
	  --input-dir ./data/04_models/yolov12/ \
	  --output-dir ./data/06_reporting/yolov12/ \
	  --loglevel "info"

run_test_suite:
	poetry run pytest
