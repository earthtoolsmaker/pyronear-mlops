.PHONY: check fix mlflow_start mlflow_stop run_yolov8_hyperparameter_search run_yolo_hyperparameter_search yolo_benchmark

check:
	isort --check .
	flake8 .
	mypy .
	black --check .

fix:
	isort .
	black .

mlflow_start:
	mlflow server --backend-store-uri runs/mlflow

mlflow_stop:
	ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9


# FIXME: deprecate the yolovN runs here and keep only one
# run_yolov8_hyperparameter_search:
# 	python ./scripts/model/yolov8/hyperparameter_search.py \
# 	  --data ./data/03_model_input/yolov8/full/datasets/data.yaml \
# 	  --output-dir ./data/04_models/yolov8/ \
# 	  --experiment-name "random_hyperparameter_search" \
# 	  --n 100 \
# 	  --loglevel "info"
#
# run_yolov9_hyperparameter_search:
# 	python ./scripts/model/yolov9/hyperparameter_search.py \
# 	  --data ./data/03_model_input/yolov8/full/datasets/data.yaml \
# 	  --output-dir ./data/04_models/yolov9/ \
# 	  --experiment-name "random_hyperparameter_search" \
# 	  --n 100 \
# 	  --loglevel "info"
#
# run_yolov10_hyperparameter_search:
# 	python ./scripts/model/yolov10/hyperparameter_search.py \
# 	  --data ./data/03_model_input/yolov8/full/datasets/data.yaml \
# 	  --output-dir ./data/04_models/yolov10/ \
# 	  --experiment-name "random_hyperparameter_search" \
# 	  --n 100 \
# 	  --loglevel "info"
#

run_yolo_hyperparameter_search:
	python ./scripts/model/yolo/hyperparameter_search.py \
	  --data ./data/03_model_input/wildfire/full/datasets/data.yaml \
	  --output-dir ./data/04_models/yolov12/ \
	  --experiment-name "random_hyperparameter_search" \
	  --model-version 12 \
	  --n 5 \
	  --loglevel "info"

# FIXME: remove
# yolov8_benchmark:
# 	python ./scripts/model/yolov8/benchmark.py \
# 	  --input-dir ./data/04_models/yolov8/ \
# 	  --output-dir ./data/06_reporting/yolov8/ \
# 	  --loglevel "info"
#
# yolov9_benchmark:
# 	python ./scripts/model/yolov8/benchmark.py \
# 	  --input-dir ./data/04_models/yolov9/ \
# 	  --output-dir ./data/06_reporting/yolov9/ \
# 	  --loglevel "info"
#
# yolov10_benchmark:
# 	python ./scripts/model/yolov8/benchmark.py \
# 	  --input-dir ./data/04_models/yolov10/ \
# 	  --output-dir ./data/06_reporting/yolov10/ \
# 	  --loglevel "info"

yolo_benchmark:
	python ./scripts/model/yolo/benchmark.py \
	  --input-dir ./data/04_models/yolov12/ \
	  --output-dir ./data/06_reporting/yolov12/ \
	  --loglevel "info"

run_test_suite:
	poetry run pytest
