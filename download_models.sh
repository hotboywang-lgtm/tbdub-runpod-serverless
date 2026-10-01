#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints /root/.cache/torch/hub/checkpoints

for model_file in \
  30_net_gen.pth \
  BFM.zip \
  DNet.pt \
  ENet.pth \
  expression.mat \
  face3d_pretrain_epoch_20.pth \
  GFPGANv1.3.pth \
  GPEN-BFR-512.pth \
  LNet.pth \
  ParseNet-latest.pth \
  RetinaFace-R50.pth \
  shape_predictor_68_face_landmarks.dat
do
  curl -fL --retry 5 --retry-delay 3 \
    "https://github.com/OpenTalker/video-retalking/releases/download/v0.0.1/${model_file}" \
    -o "checkpoints/${model_file}"
done

unzip -q checkpoints/BFM.zip -d checkpoints/BFM
curl -fL --retry 5 \
  https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth \
  -o /root/.cache/torch/hub/checkpoints/s3fd-619a316812.pth
curl -fL --retry 5 \
  https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip \
  -o /root/.cache/torch/hub/checkpoints/2DFAN4-cd938726ad.zip
rm checkpoints/BFM.zip
