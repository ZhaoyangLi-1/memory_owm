class VideoDecoder:  # pragma: no cover - never used with video.npy shards
    def __init__(self, *args, **kwargs):
        raise RuntimeError("torchcodec is not available (owm shim); store videos as `video.npy` in the shards")
