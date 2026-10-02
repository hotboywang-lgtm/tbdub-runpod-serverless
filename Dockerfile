FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CACHE_ROOT=/runpod-volume/videoretalking-cache \
    OUTPUT_MODE=s3

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl ffmpeg git libgl1 libglib2.0-0 \
      python3-dev unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/video-retalking
COPY . /app/video-retalking

RUN python -m pip install --upgrade pip "setuptools<81" wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip install runpod boto3 requests filelock scipy==1.10.1 numba==0.57.1

RUN bash download_models.sh

CMD ["python", "-u", "handler.py"]
