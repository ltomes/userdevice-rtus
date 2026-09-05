# Training image for Greyduck upscaler students (runs on ms, RTX A5000).
# traiNNer-redux deps baked in; the live repo is bind-mounted at runtime so
# code edits don't need a rebuild.
FROM pytorch/pytorch:2.12.1-cuda12.6-cudnn9-runtime

# libvips for pyvips; libGL for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvips42 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY traiNNer-redux /opt/trainner-src
RUN pip install --no-cache-dir --break-system-packages /opt/trainner-src && \
    rm -rf /opt/trainner-src

WORKDIR /workspace/traiNNer-redux

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages "onnx==1.17.0" onnxscript
RUN pip install --no-cache-dir --break-system-packages onnxruntime==1.20.1
