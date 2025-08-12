import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradio_client import Client

# ========================
# 🔹 CONFIG — insert your Gemini API key here
# ========================
GEMINI_API_KEY = "AIzaSyDpAmrLDJjDTKi7TD-IS3vqQlBAYVrUbv4"  # ⬅️ put your real key here
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY — please set it in the code")

app = FastAPI()
tts_client = Client("Ghana-NLP/Southern-Ghana-TTS-Public")

class ChatReq(BaseModel):
    prompt: str

def call_gemini_generate(prompt: str) -> str:
    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=20)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"LLM error: {resp.text}")

    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

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
