FROM python:3.12-slim

RUN pip install Flask pandas

ENV FLASK_APP=app.py

WORKDIR /code

COPY code/app.py .
COPY code/templates/* templates/
COPY code/model.pkl .
COPY code/model.py .

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]
