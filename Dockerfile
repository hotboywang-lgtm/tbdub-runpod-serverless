FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface \
    CACHE_ROOT=/runpod-volume/tbdub-cache \
    OUTPUT_MODE=s3 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/TaoLiveAIGC/TBDub.git /app/TBDub
WORKDIR /app/TBDub

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt \
    && pip install runpod boto3 filelock requests

# Bake the Student model and all runtime dependencies into the image so a
# request never waits for model downloads after entering the RunPod queue.
RUN mkdir -p checkpoints/hubert-large-ll60k \
    && hf download TaoLiveAIGC/TBDub \
      config.json null_prompt_emb.pt tbdub_student.safetensors \
      --local-dir checkpoints \
    && hf download KlingTeam/X-Dub Wan2.2_VAE.safetensors \
      --local-dir checkpoints \
    && hf download facebook/hubert-large-ll60k \
      config.json preprocessor_config.json pytorch_model.bin \
      --local-dir checkpoints/hubert-large-ll60k \
    && curl -fL \
      https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
      -o checkpoints/face_landmarker.task

COPY handler.py /app/TBDub/handler.py
COPY README.md /app/TBDub/RUNPOD_README.md

CMD ["python", "-u", "handler.py"]
