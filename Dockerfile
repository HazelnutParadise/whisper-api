FROM nvidia/cuda:12.6.3-cudnn-runtime-rockylinux8

RUN dnf update -y && \
    dnf install -y python3-pip && \
    dnf clean all
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py .
RUN mkdir -p /app/whisper_service

CMD ["python3", "app.py"]

