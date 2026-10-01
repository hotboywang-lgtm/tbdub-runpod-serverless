import hashlib
import os
import pickle
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import boto3
import requests
import runpod
import torch
from botocore.config import Config
from filelock import FileLock

from inference import PreprocessedSample, build_pipeline, preprocess_inputs, run_inference


APP_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = APP_ROOT / "checkpoints"
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/runpod-volume/tbdub-cache"))
if str(CACHE_ROOT).startswith("/runpod-volume") and not Path("/runpod-volume").exists():
    CACHE_ROOT = Path("/tmp/tbdub-cache")
SOURCE_ROOT = CACHE_ROOT / "sources"
PREPROCESS_ROOT = CACHE_ROOT / "preprocess"
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
PREPROCESS_ROOT.mkdir(parents=True, exist_ok=True)

MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "180"))
PRESIGNED_URL_TTL = int(os.getenv("PRESIGNED_URL_TTL", "86400"))
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "s3").lower()
MODEL_LOCK = FileLock(str(CACHE_ROOT / "tbdub-inference.lock"))


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _safe_id(value: str, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise ValueError(f"{field} must contain only letters, numbers, '.', '_' or '-'")
    return value


def _bounded_int(payload: dict, name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(payload.get(name, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _download(url: str, destination: Path) -> tuple[str, int]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http/https URLs are accepted")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    with requests.get(url, stream=True, timeout=(15, DOWNLOAD_TIMEOUT), allow_redirects=True) as response:
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length and int(length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Remote file exceeds MAX_DOWNLOAD_BYTES")
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError("Remote file exceeds MAX_DOWNLOAD_BYTES")
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest(), total


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _normalize_video(source: Path, destination: Path, duration: float) -> None:
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", "fps=25,scale=trunc(iw/2)*2:trunc(ih/2)*2,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination),
        ]
    )


def _normalize_audio(source: Path, destination: Path, duration: float) -> None:
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-t", f"{duration:.3f}", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(destination),
        ]
    )


def _finalize_video(source: Path, destination: Path) -> None:
    _run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-vf", "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart", str(destination),
        ]
    )


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version=os.getenv("S3_SIGNATURE_VERSION", "s3v4")),
    )


def _publish(path: Path, job_id: str) -> tuple[str, str]:
    if OUTPUT_MODE != "s3":
        raise RuntimeError("This endpoint requires OUTPUT_MODE=s3")
    bucket = os.environ["S3_BUCKET"]
    prefix = os.getenv("S3_PREFIX", "tbdub").strip("/")
    key = f"{prefix}/outputs/{job_id}-{uuid.uuid4().hex[:10]}.mp4"
    client = _s3_client()
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=PRESIGNED_URL_TTL
    )
    return key, url


MODEL_ARGS = SimpleNamespace(
    dit_checkpoint=[str(CHECKPOINT_ROOT / "tbdub_student.safetensors")],
    vae_checkpoint=str(CHECKPOINT_ROOT / "Wan2.2_VAE.safetensors"),
    prompt_embedding=str(CHECKPOINT_ROOT / "null_prompt_emb.pt"),
    hubert_checkpoint=str(CHECKPOINT_ROOT / "hubert-large-ll60k"),
    mediapipe_model=str(CHECKPOINT_ROOT / "face_landmarker.task"),
    preprocess_report=None,
    device="cuda:0",
    cpu_offload=True,
    num_frames=None,
    audio_feat_window_size=0,
)


def _load_pipeline():
    started = time.perf_counter()
    pipeline, prompt_embedding = build_pipeline(MODEL_ARGS)
    return pipeline, prompt_embedding, time.perf_counter() - started


PIPELINE, PROMPT_EMBEDDING, MODEL_LOAD_SECONDS = _load_pipeline()


def _load_or_prepare_avatar(avatar_id: str, source_video: Path, audio_path: Path, refresh: bool):
    cache_path = PREPROCESS_ROOT / f"{avatar_id}.pkl"
    cache_hit = cache_path.exists() and not refresh
    with FileLock(str(PREPROCESS_ROOT / f"{avatar_id}.lock")):
        if cache_path.exists() and not refresh:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            sample = PreprocessedSample(
                raw_video=cached["raw_video"],
                reference_video=cached["reference_video"],
                bboxes=cached["bboxes"],
                audio_path=str(audio_path),
            )
            return sample, True, 0.0

        started = time.perf_counter()
        sample = preprocess_inputs(str(source_video), str(audio_path), MODEL_ARGS)
        temp_path = cache_path.with_suffix(".tmp.pkl")
        with temp_path.open("wb") as handle:
            pickle.dump(
                {
                    "raw_video": sample.raw_video,
                    "reference_video": sample.reference_video,
                    "bboxes": sample.bboxes,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temp_path.replace(cache_path)
        return sample, cache_hit, time.perf_counter() - started


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    job_id = _safe_id(payload.get("job_id", uuid.uuid4().hex), "job_id")
    avatar_id = _safe_id(payload.get("avatar_id", "default"), "avatar_id")
    video_url = payload.get("source_video_url")
    audio_url = payload.get("audio_url")
    if not video_url or not audio_url:
        raise ValueError("source_video_url and audio_url are required")

    max_duration = _bounded_int(payload, "max_duration_seconds", 30, 1, 180)
    seed = _bounded_int(payload, "seed", 42, 0, 2_147_483_647)
    refresh_avatar = bool(payload.get("refresh_avatar", False))

    started = time.perf_counter()
    work = Path("/tmp") / f"tbdub-{job_id}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=False)
    audio_download = work / "audio-download"
    normalized_audio = work / "audio.wav"
    source_download = work / "source-download"
    generated_dir = work / "generated"
    final_video = work / "final.mp4"

    try:
        download_started = time.perf_counter()
        _, audio_bytes = _download(audio_url, audio_download)
        audio_duration = min(_probe_duration(audio_download), float(max_duration))
        if audio_duration <= 0:
            raise ValueError("audio duration must be greater than zero")
        _normalize_audio(audio_download, normalized_audio, audio_duration)

        cached_source = SOURCE_ROOT / f"{avatar_id}.mp4"
        source_cache_hit = cached_source.exists() and not refresh_avatar
        with FileLock(str(SOURCE_ROOT / f"{avatar_id}.lock")):
            if refresh_avatar or not cached_source.exists():
                _download(video_url, source_download)
                source_duration = min(_probe_duration(source_download), 600.0)
                if source_duration <= 0:
                    raise ValueError("source video duration must be greater than zero")
                temp_source = cached_source.with_suffix(".tmp.mp4")
                _normalize_video(source_download, temp_source, source_duration)
                temp_source.replace(cached_source)
            else:
                source_cache_hit = True
        download_seconds = time.perf_counter() - download_started

        sample, avatar_cache_hit, avatar_prepare_seconds = _load_or_prepare_avatar(
            avatar_id, cached_source, normalized_audio, refresh_avatar
        )

        inference_args = SimpleNamespace(
            start_frame=0,
            inference_mode="student",
            student_first_clip_padding=True,
            num_student_steps=2,
            num_inference_steps=30,
            sigma_shift=1.0,
            motion_from_latents=True,
            seed=seed,
            per_chunk_audio=False,
            ref_cfg_scale=2.0,
            audio_cfg_scale=6.0,
            output_dir=str(generated_dir),
            save_comparison=False,
            cropped_input=False,
        )

        inference_started = time.perf_counter()
        with MODEL_LOCK:
            generated = run_inference(
                PIPELINE, PROMPT_EMBEDDING, sample, job_id, inference_args
            )
        inference_seconds = time.perf_counter() - inference_started

        _finalize_video(generated, final_video)
        object_key, output_url = _publish(final_video, job_id)
        return {
            "status": "completed",
            "job_id": job_id,
            "avatar_id": avatar_id,
            "video_url": output_url,
            "object_key": object_key,
            "audio_duration_seconds": round(audio_duration, 3),
            "audio_bytes": audio_bytes,
            "source_cache_hit": source_cache_hit,
            "avatar_cache_hit": avatar_cache_hit,
            "model": "TaoLiveAIGC/TBDub Student",
            "generation_settings": {
                "inference_mode": "student",
                "student_steps": 2,
                "sigma_shift": 1.0,
                "motion_from_latents": True,
                "cpu_offload": True,
                "seed": seed,
            },
            "model_load_seconds": round(MODEL_LOAD_SECONDS, 3),
            "download_seconds": round(download_seconds, 3),
            "avatar_prepare_seconds": round(avatar_prepare_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "processing_time_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
