# Kimodo for batch generation, on a CUDA 12.8 PyTorch (Blackwell GPUs such as the RTX 5090 need
# it; Kimodo's own image is on CUDA 12.6). Build context: a Kimodo checkout at the commit `setup`
# pins, with this folder's files copied next to it (see ../README.md).
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
# a broken pip `cmake` shim would shadow the system one (as in Kimodo's own Dockerfile)
RUN rm -f /usr/local/bin/cmake

WORKDIR /workspace
COPY docker_requirements.txt setup.py pyproject.toml /workspace/
COPY kimodo /workspace/kimodo
COPY MotionCorrection /workspace/MotionCorrection

# Kimodo's lockfile without the interactive demo's viewer (batch generation does not use it)
RUN grep -v "kimodo-viser" docker_requirements.txt > requirements.txt \
    && SKIP_MOTION_CORRECTION_IN_SETUP=1 python -m pip install -r requirements.txt

ENV HF_HOME=/hf
CMD ["bash"]
