FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
# WhisperX currently shells out to the ffmpeg CLI when decoding audio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY whisper_service ./whisper_service
CMD ["python3", "-m", "whisper_service.app"]
