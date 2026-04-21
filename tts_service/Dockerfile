FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY tts_service/requirements.txt ./tts_service-requirements.txt
RUN pip install --no-cache-dir -r ./tts_service-requirements.txt
COPY tts_service ./tts_service

ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "uvicorn", "tts_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
