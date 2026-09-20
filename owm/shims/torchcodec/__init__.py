"""Import shim: videosaur's data/pipelines.py does `from torchcodec.decoders import VideoDecoder` at import time,
but the torchcodec wheel does not load against this torch build. Our shards carry `video.npy` arrays, so the
decoder is never called. This directory is put on PYTHONPATH only by the owm launchers."""
