from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Allow the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure sessions folder exists
SESSION_DIR = "sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

# 1. Load the AI Model
print("Loading BanglaT5 Model... Please wait.")
t5_model_name = "csebuetnlp/banglat5"
try:
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    t5_model = None

# 2. Data Structures


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

# 3. AI Translation Endpoint


@app.post("/process_text")
async def process_text(req: TextRequest):
    if t5_model and t5_tokenizer:
        try:
            input_ids = t5_tokenizer(
                req.message, return_tensors="pt").input_ids
            outputs = t5_model.generate(input_ids, max_length=200)
            response_text = t5_tokenizer.decode(
                outputs[0], skip_special_tokens=True)
        except Exception as e:
            response_text = f"Error during translation: {str(e)}"
    else:
        response_text = f"Standard Bangla (Echo for testing): {req.message}"

    return {"response": response_text}

# 4. Save Chat History Endpoint


@app.post("/save_session")
async def save_session(data: SessionData):
    if not data.filename:
        # Create a new filename if this is a new chat
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data.filename = f"chat_{timestamp}.json"

    filepath = os.path.join(SESSION_DIR, data.filename)

    # Save to JSON
    with open(filepath, "w", encoding="utf-8") as f:
        messages_dict = [{"role": m.role, "content": m.content}
                         for m in data.messages]
        json.dump({
            "title": data.title,
            "messages": messages_dict
        }, f, ensure_ascii=False, indent=4)

    return {"status": "success", "filename": data.filename}

# 5. Get Sidebar History Endpoint


@app.get("/get_sessions")
async def get_sessions():
    sessions = []
    if not os.path.exists(SESSION_DIR):
        return sessions

    for file in sorted(os.listdir(SESSION_DIR), reverse=True):  # Newest first
        if file.endswith(".json"):
            try:
                with open(os.path.join(SESSION_DIR, file), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "filename": file,
                        "title": data.get("title", "New Chat")
                    })
            except Exception:
                pass
    return sessions

# 6. Load Specific Chat Endpoint


@app.get("/get_session/{filename}")
async def get_session(filename: str):
    filepath = os.path.join(SESSION_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "File not found"}
