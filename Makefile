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
