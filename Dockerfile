FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CACHE_ROOT=/runpod-volume/videoretalking-cache \
    OUTPUT_MODE=s3

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl ffmpeg git libgl1 libglib2.0-0 \
      python3 python3-dev python3-pip unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/video-retalking
COPY . /app/video-retalking

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install \
      torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install runpod boto3 requests filelock scipy==1.10.1 numba==0.57.1

RUN mkdir -p checkpoints /root/.cache/torch/hub/checkpoints \
    && for f in 30_net_gen.pth BFM.zip DNet.pt ENet.pth expression.mat \
      face3d_pretrain_epoch_20.pth GFPGANv1.3.pth GPEN-BFR-512.pth \
      LNet.pth ParseNet-latest.pth RetinaFace-R50.pth \
      shape_predictor_68_face_landmarks.dat; do \
        curl -fL --retry 5 --retry-delay 3 \
          "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/${f}" \
          -o "checkpoints/${f}"; \
      done \
    && unzip -q checkpoints/BFM.zip -d checkpoints/BFM \
    && curl -fL --retry 5 \
      https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
      -o /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth \
    && curl -fL --retry 5 \
      https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip \
      -o /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip \
    && rm checkpoints/BFM.zip

CMD ["python3", "-u", "handler.py"]
