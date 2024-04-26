.PHONY: check fix mlflow_start mlflow_stop run_yolov8_hyperparameter_search
	run_yolov9_hyperparameter_search yolov8_benchmark yolov9_benchmark

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


run_yolov8_hyperparameter_search:
	python ./scripts/model/yolov8/hyperparameter_search.py \
	  --data ./data/03_model_input/yolov8/full/datasets/data.yaml \
	  --output-dir ./data/04_models/yolov8/ \
	  --experiment-name "random_hyperparameter_search" \
	  --n 10 \
	  --loglevel "info"

run_yolov9_hyperparameter_search:
	python ./scripts/model/yolov9/hyperparameter_search.py \
	  --data ./data/03_model_input/yolov8/full/datasets/data.yaml \
	  --output-dir ./data/04_models/yolov9/ \
	  --experiment-name "random_hyperparameter_search" \
	  --n 10 \
	  --loglevel "info"

yolov8_benchmark:
	python ./scripts/model/yolov8/benchmark.py \
	  --input-dir ./data/04_models/yolov8/ \
	  --output-dir ./data/06_reporting/yolov8/ \
	  --loglevel "info"

yolov9_benchmark:
	python ./scripts/model/yolov8/benchmark.py \
	  --input-dir ./data/04_models/yolov9/ \
	  --output-dir ./data/06_reporting/yolov9/ \
	  --loglevel "info"
