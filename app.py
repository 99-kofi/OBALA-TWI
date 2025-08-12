import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env

GEMINI_API_URL = os.getenv("GEMINI_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or not GEMINI_API_URL:
    raise RuntimeError("Missing GEMINI_API_KEY or GEMINI_API_URL in environment")

app = FastAPI()
tts_client = Client("Ghana-NLP/Southern-Ghana-TTS-Public")

class ChatReq(BaseModel):
    prompt: str

def call_gemini_generate(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gemini-2.0-flash",
        "prompt": prompt,
        "temperature": 0.2,
        "max_output_tokens": 512
    }
    resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM error: {resp.text}")
    return resp.json().get("output_text", "")

def translate_to_twi(english_text: str) -> str:
    prompt = f"Translate this into Asante Twi:\n{english_text}"
    return call_gemini_generate(prompt)

@app.post("/chat")
def chat(req: ChatReq):
    english_reply = call_gemini_generate(req.prompt)
    twi_reply = translate_to_twi(english_reply)
    tts_result = tts_client.predict(
        text=twi_reply,
        lang="Asante Twi",
        speaker="Female",
        api_name="/predict"
    )
    return {
        "english": english_reply,
        "twi": twi_reply,
        "tts": tts_result
    }
