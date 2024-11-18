setup:
	python3 -m venv venv
	. venv/bin/activate && pip install poetry
	poetry lock
	poetry install
