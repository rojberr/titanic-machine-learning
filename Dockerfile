FROM python:3.9-slim

WORKDIR /app

COPY /code/predict.py .

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org/ | python3 -

#RUN ~/.local/share/pypoetry/venv/bin/poetry config virtualenvs.create false --local \
#    && ~/.local/share/pypoetry/venv/bin/poetry install --no-interaction --no-ansi

CMD ["python", "code/train.py"]
