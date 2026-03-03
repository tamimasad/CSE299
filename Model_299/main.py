from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
import tempfile
import torch
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    WhisperProcessor, 
    WhisperForConditionalGeneration
)
from pydub import AudioSegment
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Models

t5_model_name = "csebuetnlp/banglat5"
whisper_model_name = "openai/whisper-small"

try:
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name, use_fast=False)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)
    
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
    print("Models loaded successfully!")
except Exception as e:
    print(f"Model Loading Error: {e}")

class Message(BaseModel):
    role: str
    content: str

class SessionData(BaseModel):
    title: str
    messages: List[Message]
    filename: Optional[str] = None

class TextRequest(BaseModel):
    message: str
    model: str

def generate_standard_bangla(input_text: str):
    if not t5_model or not t5_tokenizer:
        return input_text
    
    # Prefix for BanglaT5 

    text = "standardize: " + input_text
    input_ids = t5_tokenizer(text, return_tensors="pt").input_ids.to(device)
    
    outputs = t5_model.generate(
        input_ids, 
        max_length=128, 
        num_beams=5,                 
        no_repeat_ngram_size=3,      
        repetition_penalty=1.3,     
        length_penalty=1.0,
        early_stopping=True
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/process_text")
async def process_text(req: TextRequest):
    try:
        response_text = generate_standard_bangla(req.message)
        return {"response": response_text}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}

@app.post("/process_voice")
async def process_voice(audio: UploadFile = File(...), model: str = Form("standard")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    try:
        audio_segment = AudioSegment.from_file(temp_audio_path)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0

        input_features = whisper_processor(samples, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="bn", task="transcribe")
        
        predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcribed_text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        standard_text = generate_standard_bangla(transcribed_text)
        return {"transcribed_text": transcribed_text, "response": standard_text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.post("/save_session")
async def save_session(data: SessionData):
    if not data.filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data.filename = f"chat_{timestamp}.json"
    
    filepath = os.path.join(SESSION_DIR, data.filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"title": data.title, "messages": [m.dict() for m in data.messages]}, f, ensure_ascii=False, indent=4)
    return {"status": "success", "filename": data.filename}

@app.get("/get_sessions")
async def get_sessions():
    sessions = []
    if not os.path.exists(SESSION_DIR): return sessions
    for file in sorted(os.listdir(SESSION_DIR), reverse=True): 
        if file.endswith(".json"):
            with open(os.path.join(SESSION_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append({"filename": file, "title": data.get("title", "New Chat")})
    return sessions

@app.get("/get_session/{filename}")
async def get_session(filename: str):
    filepath = os.path.join(SESSION_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Not found"}