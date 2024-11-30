.PHONY: clean check-python-version train predict

PYTHONVERSION = $(shell python3 --version 2>&1 | awk '{print $$2}')
VENV = ./venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Check Python version
check-python-version:
	@echo "Checking Python version..."
	@if ! python3 -c "import sys; \
		version = tuple(map(int, sys.version.split()[0].split('.'))); \
		sys.exit(not ((3, 8, 0) <= version <= (3, 11, 9)))"; then \
		echo "Error: Python version must be between 3.8 and 3.11.9. Detected: $(PYTHONVERSION)"; \
		exit 1; \
	fi

$(VENV)/bin/activate: requirements.txt check-python-version
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

venv: clean check-python-version
	@echo "Creating a new virtual environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate
	@echo "Virtual environment rejuvenated."

train: $(VENV)/bin/activate
	$(PYTHON) src/main.py train

predict: $(VENV)/bin/activate
	$(PYTHON) src/main.py predict

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
