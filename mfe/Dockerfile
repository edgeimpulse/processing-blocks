# syntax = docker/dockerfile:experimental
FROM python:3.7.5-stretch

WORKDIR /app

# Python dependencies
COPY requirements.txt ./
RUN pip3 --no-cache-dir install -r requirements.txt

COPY . ./

EXPOSE 4446

CMD python3 -u dsp-server.py
