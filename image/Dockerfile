# syntax = docker/dockerfile:experimental
FROM python:3.7.5-stretch

WORKDIR /app

# Python dependencies
COPY requirements-blocks.txt ./
RUN pip3 --no-cache-dir install -r requirements-blocks.txt

COPY third_party /third_party
COPY . ./

EXPOSE 4446

CMD python3 -u dsp-server.py
