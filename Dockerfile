FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

COPY src /workspace/src
COPY scripts /workspace/scripts

ENV PYTHONPATH=/workspace/src

EXPOSE 8501

CMD ["bash"]
