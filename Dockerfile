FROM python:3.11-slim

# soundfile(libsndfile1), librosa 코덱(ffmpeg), sounddevice 임포트(libportaudio2) 필요.
# 마이크 모드는 비목표이나 sounddevice는 임포트 시점에 libportaudio2를 로드하므로 포함.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TFHUB_CACHE_DIR=/opt/tfhub_cache
ENV TF_CPP_MIN_LOG_LEVEL=2

# requirements 먼저 COPY → pip install 레이어 캐시 유지.
# 소스 코드 변경 시 이 레이어는 재실행되지 않는다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# YAMNet 캐시: scripts/만 먼저 COPY 후 다운로드.
# verify_inference.py 는 src를 임포트하지 않으므로 src COPY 불필요.
COPY scripts/ scripts/
RUN python scripts/verify_inference.py

# 소스 코드 — 가장 자주 바뀌므로 마지막에 배치.
COPY src/ src/
COPY config/ config/
COPY tests/ tests/

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["--help"]
