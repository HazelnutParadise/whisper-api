from fastapi import FastAPI, UploadFile, File, HTTPException, Form
import whisper
import os
import uuid
import shutil
from typing import Any

app = FastAPI()
model: whisper.Whisper = whisper.load_model("turbo")  # You can use "medium" or "large" if GPU is available

UPLOAD_FOLDER: str = "./whisper_service"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile = File(...), model_name: str = Form("whisper-1"))-> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    filename: str = f"{uuid.uuid4().hex}.wav"
    filepath: str = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result: dict = model.transcribe(filepath)
    
    # Clean up temporary file
    os.remove(filepath)

    return {"text": result["text"].strip()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
