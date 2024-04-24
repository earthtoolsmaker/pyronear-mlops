.PHONY: check fix

check:
	isort --check .
	flake8 .
	mypy .
	black --check .

fix:
	isort .
	black .

mlflow_start:
	# mlflow ui --port 5000
	mlflow server --backend-store-uri runs/mlflow

mlflow_stop:
	ps aux | grep 'mlflow' | grep -v 'grep' | awk '{print $2}' | xargs kill -9


run_hyperparameter_search:
	python ./scripts/model/yolov8/hyperparameter_search.py \
	  --data ./data/03_model_input/yolov8/small/datasets/data.yaml \
	  --output-dir ./data/04_models/yolov8/ \
	  --experiment-name "random_hyperparameter_search" \
	  --n 10 \
	  --loglevel "info"
