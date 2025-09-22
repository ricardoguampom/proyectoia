# =========================
# path: Dockerfile
# =========================
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates libgl1 libglib2.0-0 fonts-dejavu-core && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -U pip && pip install -r requirements.txt

COPY tools/ ./tools/
ENTRYPOINT ["python", "tools/describe_objects.py"]
