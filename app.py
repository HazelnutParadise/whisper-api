from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import whisper
import os
import uuid
import shutil
import logging
from typing import Any
from contextlib import asynccontextmanager

# Delay loading the whisper model until startup so we can verify ffmpeg first
model = None

UPLOAD_FOLDER: str = "./whisper_service"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure ffmpeg is available — whisper's audio loader uses ffmpeg subprocess
    if shutil.which("ffmpeg") is None:
        logging.error("ffmpeg not found in PATH. Please install ffmpeg in the container or host.")
        raise RuntimeError("ffmpeg not found. Install ffmpeg in the container or host.")
    global model
    # Load model after verifying dependencies
    model = whisper.load_model("turbo")  # You can use "medium" or "large" if GPU is available
    try:
        yield
    finally:
        # Optional cleanup on shutdown
        model = None


app = FastAPI(lifespan=lifespan)

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model_name: str = Form("whisper-1"))-> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filename: str = f"{uuid.uuid4().hex}.wav"
    filepath: str = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result: dict = model.transcribe(filepath)
    
    # Clean up temporary file
    os.remove(filepath)

    return {"text": result["text"].strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
