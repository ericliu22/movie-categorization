FROM python:3.10-slim

WORKDIR /usr/src/app

COPY . .

RUN pip install poetry

RUN poetry install

CMD ["python", "src/main.py"]
