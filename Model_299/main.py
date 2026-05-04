import os
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import tempfile
import torch
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    AutoProcessor,
    AutoModelForImageTextToText
)
from pydub import AudioSegment
import numpy as np

load_dotenv()

app = FastAPI()

# Enable CORS for frontend communication
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

# --- Model Configurations ---
t5_model_name = "./fine_tuned_banglat5_final"
whisper_model_name = "openai/whisper-small"
m2m100_model_name = "./fine_tuned_m2m100"
gemma_model_name = "./gemma4_dialect_results/final_adapter"

# --- Pydantic Models for API Requests ---


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


# --- Global Model Initialization ---
t5_tokenizer = t5_model = None
whisper_processor = whisper_model = None
m2m_tokenizer = m2m_model = None
gemma_processor = gemma_model = None

try:
    print("Loading models... please wait.")

    # Load BanglaT5
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name, use_fast=False)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name).to(device)

    # Load Whisper
    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_name).to(device)

    # Load M2M100
    m2m_tokenizer = M2M100Tokenizer.from_pretrained(m2m100_model_name)
    m2m_model = M2M100ForConditionalGeneration.from_pretrained(
        m2m100_model_name).to(device)

    # Load Gemma 4 E2B
    gemma_processor = AutoProcessor.from_pretrained(gemma_model_name)
    gemma_model = AutoModelForImageTextToText.from_pretrained(
        gemma_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)

    print("All Models loaded successfully!")
except Exception as e:
    print(f"Model Loading Error: {e}")

# --- Core Translation Logic ---


def generate_standard_bangla(input_text: str, model_choice: str):
    # 1. Gemma Logic
    if model_choice == "gemma" and gemma_model and gemma_processor:
        prompt = f"Transform the following dialect Bengali text to standard Bengali: {input_text}"
        inputs = gemma_processor(text=prompt, return_tensors="pt").to(device)
        generated_ids = gemma_model.generate(**inputs, max_new_tokens=128)
        # Decoding and removing the prompt from output
        full_text = gemma_processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return full_text.replace(prompt, "").strip()

    # 2. M2M100 Logic
    elif model_choice == "m2m100" and m2m_model and m2m_tokenizer:
        m2m_tokenizer.src_lang = "bn"
        encoded_bn = m2m_tokenizer(input_text, return_tensors="pt").to(device)
        generated_tokens = m2m_model.generate(
            **encoded_bn,
            forced_bos_token_id=m2m_tokenizer.get_lang_id("bn"),
            max_length=64
        )
        return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    # 3. Default/BanglaT5 Logic
    else:
        if not t5_model or not t5_tokenizer:
            return input_text
        text = "paraphrase: " + input_text
        input_ids = t5_tokenizer(
            text, return_tensors="pt").input_ids.to(device)
        outputs = t5_model.generate(
            input_ids,
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=2,
            repetition_penalty=3.5,
            length_penalty=1.0,
            early_stopping=True
        )
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- FastAPI Endpoints ---


@app.post("/process_text")
async def process_text(req: TextRequest):
    try:
        standard_text = generate_standard_bangla(req.message, req.model)
        return {"response": standard_text}
    except Exception as e:
        return {"error": str(e)}


@app.post("/process_voice")
async def process_voice(audio: UploadFile = File(...), model: str = Form(...)):
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(await audio.read())
            temp_audio_path = tmp.name

        # Convert and Process Audio
        audio_seg = AudioSegment.from_file(temp_audio_path)
        audio_array = np.array(
            audio_seg.get_array_of_samples(), dtype=np.float32) / 32768.0

        inputs = whisper_processor(
            audio_array, sampling_rate=16000, return_tensors="pt").to(device)
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(
            language="bengali", task="transcribe")
        ids = whisper_model.generate(
            inputs.input_features, forced_decoder_ids=forced_decoder_ids)
        transcribed_text = whisper_processor.batch_decode(
            ids, skip_special_tokens=True)[0]

        standard_text = generate_standard_bangla(transcribed_text, model)
        return {"transcribed_text": transcribed_text, "response": standard_text}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.post("/save_session")
async def save_session(data: SessionData):
    if not data.filename:
        data.filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(SESSION_DIR, data.filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({
            "title": data.title,
            "messages": [m.model_dump() for m in data.messages]
        }, f, ensure_ascii=False, indent=4)
    return {"status": "success", "filename": data.filename}


@app.get("/get_sessions")
async def get_sessions():
    sessions = []
    if not os.path.exists(SESSION_DIR):
        return sessions
    for file in sorted(os.listdir(SESSION_DIR), reverse=True):
        if file.endswith(".json"):
            with open(os.path.join(SESSION_DIR, file), "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions.append(
                    {"filename": file, "title": data.get("title", "New Chat")})
    return sessions


@app.get("/get_session/{filename}")
async def get_session(filename: str):
    filepath = os.path.join(SESSION_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
