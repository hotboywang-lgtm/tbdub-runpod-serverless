# TBDub Student RunPod Serverless

Independent RunPod Serverless worker for `TaoLiveAIGC/TBDub` Student. It does
not replace or modify the existing MuseTalk or LatentSync endpoints.

## Request

```json
{
  "input": {
    "job_id": "example-001",
    "avatar_id": "speaker-001",
    "source_video_url": "https://example.com/source.mp4",
    "audio_url": "https://example.com/audio.wav",
    "max_duration_seconds": 5,
    "seed": 42,
    "refresh_avatar": false
  },
  "policy": {
    "ttl": 3600000
  }
}
```

The worker immediately downloads expiring input URLs, normalizes the source to
25 fps BT.709, caches face preprocessing by `avatar_id`, runs the two-step
Student model with CPU offload for 24 GB GPUs, uploads the MP4 to S3-compatible
object storage, and returns a presigned `video_url`.

## Required environment variables

- `S3_ENDPOINT_URL`
- `S3_REGION`
- `S3_BUCKET`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`
- `S3_PREFIX=tbdub`
- `S3_SIGNATURE_VERSION=s3`
- `OUTPUT_MODE=s3`
- `PRESIGNED_URL_TTL=86400`

Recommended RunPod settings: RTX 4090 24 GB, max workers 1, active workers 0,
idle timeout 30 seconds, FlashBoot enabled.

The image pins protobuf 4.x for MediaPipe compatibility.
