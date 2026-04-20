FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ARG HIGGS_AUDIO_COMMIT=8b1539a02d5764317724a904a344a0f1be8a736e

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/boson-ai/higgs-audio.git \
    && cd higgs-audio \
    && git checkout "${HIGGS_AUDIO_COMMIT}" \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

WORKDIR /app
COPY tts_service/requirements.txt ./tts_service-requirements.txt
RUN pip install --no-cache-dir -r ./tts_service-requirements.txt
COPY tts_service ./tts_service

ENV HF_HUB_DISABLE_XET=1
ENV HIGGS_AUDIO_MODEL=bosonai/higgs-audio-v2-generation-3B-base
ENV HIGGS_AUDIO_TOKENIZER=bosonai/higgs-audio-v2-tokenizer
ENV HIGGS_AUDIO_VOICE_PROMPTS_DIR=/opt/higgs-audio/examples/voice_prompts
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "uvicorn", "tts_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
