FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt
