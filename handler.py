import hashlib
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3
import requests
import runpod
from botocore.config import Config
from filelock import FileLock


APP_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", "/runpod-volume/videoretalking-cache"))
if str(CACHE_ROOT).startswith("/runpod-volume") and not Path("/runpod-volume").exists():
    CACHE_ROOT = Path("/tmp/videoretalking-cache")
SOURCE_ROOT = CACHE_ROOT / "sources"
PREPROCESS_ROOT = CACHE_ROOT / "preprocess"
SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
PREPROCESS_ROOT.mkdir(parents=True, exist_ok=True)

# The upstream implementation writes reusable landmark/3DMM caches below ./temp.
temp_path = APP_ROOT / "temp"
if not temp_path.exists():
    temp_path.symlink_to(PREPROCESS_ROOT, target_is_directory=True)

MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
DOWNLOAD_TIMEOUT = int(os.getenv("DOWNLOAD_TIMEOUT", "180"))
PRESIGNED_URL_TTL = int(os.getenv("PRESIGNED_URL_TTL", "86400"))
OUTPUT_MODE = os.getenv("OUTPUT_MODE", "s3").lower()
INFERENCE_LOCK = FileLock(str(CACHE_ROOT / "videoretalking-inference.lock"))


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "no command output").strip()
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{stderr[-8000:]}"
        )


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
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _normalize_video(source: Path, destination: Path, duration: float) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-t", f"{duration:.3f}",
        "-vf", "fps=25,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-an", "-c:v", "mpeg4", "-q:v", "2",
        "-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709",
        "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", str(destination),
    ])


def _normalize_audio(source: Path, destination: Path, duration: float) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-t", f"{duration:.3f}", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(destination),
    ])


def _finalize_video(source: Path, destination: Path) -> None:
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
        "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p",
        "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(destination),
    ])


def _s3_client():
    return boto3.client(
        "s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        region_name=os.getenv("S3_REGION", "auto"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version=os.getenv("S3_SIGNATURE_VERSION", "s3v4")),
    )


def _publish(path: Path, job_id: str) -> tuple[str, str]:
    if OUTPUT_MODE != "s3":
        raise RuntimeError("This endpoint requires OUTPUT_MODE=s3")
    bucket = os.environ["S3_BUCKET"]
    prefix = os.getenv("S3_PREFIX", "videoretalking").strip("/")
    key = f"{prefix}/outputs/{job_id}-{uuid.uuid4().hex[:10]}.mp4"
    client = _s3_client()
    client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": "video/mp4"})
    url = client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=PRESIGNED_URL_TTL
    )
    return key, url


def handler(job: dict) -> dict:
    payload = job.get("input") or {}
    job_id = _safe_id(payload.get("job_id", uuid.uuid4().hex), "job_id")
    avatar_id = _safe_id(payload.get("avatar_id", "default"), "avatar_id")
    video_url = payload.get("source_video_url")
    audio_url = payload.get("audio_url")
    if not video_url or not audio_url:
        raise ValueError("source_video_url and audio_url are required")
    max_duration = _bounded_int(payload, "max_duration_seconds", 5, 1, 60)
    refresh_avatar = bool(payload.get("refresh_avatar", False))

    started = time.perf_counter()
    work = Path("/tmp") / f"videoretalking-{job_id}-{uuid.uuid4().hex[:8]}"
    work.mkdir(parents=True, exist_ok=False)
    audio_download = work / "audio-download"
    audio_wav = work / "audio.wav"
    source_download = work / "source-download"
    raw_output = work / "raw.mp4"
    final_output = work / "final.mp4"

    try:
        download_started = time.perf_counter()
        _, audio_bytes = _download(audio_url, audio_download)
        audio_duration = min(_probe_duration(audio_download), float(max_duration))
        _normalize_audio(audio_download, audio_wav, audio_duration)

        cached_source = SOURCE_ROOT / f"{avatar_id}.mp4"
        source_cache_hit = cached_source.exists() and not refresh_avatar
        with FileLock(str(SOURCE_ROOT / f"{avatar_id}.lock")):
            if refresh_avatar or not cached_source.exists():
                _download(video_url, source_download)
                source_duration = min(_probe_duration(source_download), 600.0)
                temp_source = cached_source.with_suffix(".tmp.mp4")
                _normalize_video(source_download, temp_source, source_duration)
                temp_source.replace(cached_source)
                source_cache_hit = False
        download_seconds = time.perf_counter() - download_started

        inference_started = time.perf_counter()
        with INFERENCE_LOCK:
            _run([
                "python3", "inference.py", "--face", str(cached_source),
                "--audio", str(audio_wav), "--outfile", str(raw_output),
                "--tmp_dir", f"job-{job_id}", "--LNet_batch_size", "16",
                "--face_det_batch_size", "8",
            ], cwd=APP_ROOT)
        inference_seconds = time.perf_counter() - inference_started

        _finalize_video(raw_output, final_output)
        object_key, output_url = _publish(final_output, job_id)
        return {
            "status": "completed",
            "job_id": job_id,
            "avatar_id": avatar_id,
            "video_url": output_url,
            "object_key": object_key,
            "audio_duration_seconds": round(audio_duration, 3),
            "audio_bytes": audio_bytes,
            "source_cache_hit": source_cache_hit,
            "model": "OpenTalker/VideoReTalking",
            "settings": {"fps": 25, "img_size": 384, "expression": "neutral"},
            "download_seconds": round(download_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "processing_time_seconds": round(time.perf_counter() - started, 3),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
