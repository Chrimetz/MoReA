FROM python:3.9-slim-buster

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

COPY ./app /app/app

EXPOSE 8000

CMD [ "python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000" ]
