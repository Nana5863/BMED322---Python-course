FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

WORKDIR /workspaces/Python-Course

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt
