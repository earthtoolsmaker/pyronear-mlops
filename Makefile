.PHONY: check fix

check:
	isort --check .
	flake8 .
	mypy .
	black --check .

fix:
	isort .
	black .
