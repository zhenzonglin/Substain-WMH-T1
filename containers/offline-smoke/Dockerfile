FROM nvidia/cuda:12.4.1-base-ubuntu20.04

# 仅为断网验收提供与 Ubuntu-20.04 core wheel 一致的 CPython 3.8。
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       python3 python3-venv libgomp1 libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
