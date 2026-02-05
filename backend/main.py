# pip install "google-genai>=0.1.0" fastapi uvicorn python-multipart pydantic

import json
import os
from fastapi import FastAPI, UploadFile, Form, File
from pydantic import BaseModel
from google import genai
from google.genai import types
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ---------------- CONFIG ----------------
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class TextRequest(BaseModel):
    text: str
    lang: str = "en"

class SafetyRequest(BaseModel):
    question: str
    lang: str = "en"

# ---------------- UTIL ----------------
def parse_gemini_json(text_response: str):
    clean = text_response.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.endswith("```"):
        clean = clean[:-3]
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw": text_response}

# =========================================================
# CHECK TEXT — MEDICAL MISINFORMATION
# =========================================================
@app.post("/check_text")
def check_text(req: TextRequest):
    prompt = f"""
You are a medical misinformation detection assistant for the general public in India.

TASKS:
1. Classify the claim as:
   - misinformation
   - misleading
   - reliable
   - unknown
2. Explain why.
3. Describe potential harm.
4. Provide correct information.
5. Suggest safe next steps.

RULES:
- Do NOT prescribe medicines.
- Do NOT suggest dosages.
- Use simple language.
- Be culturally appropriate.
- Do not shame the user.

Respond STRICTLY in this language: {req.lang}
Respond STRICTLY in valid JSON.

Claim:
{req.text}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["misinformation", "misleading", "reliable", "unknown"]
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    },
                    "why": {"type": "string"},
                    "potential_harm": {"type": "string"},
                    "correct_information": {"type": "string"},
                    "what_to_do": {"type": "string"}
                },
                "required": [
                    "verdict",
                    "confidence",
                    "why",
                    "potential_harm",
                    "correct_information",
                    "what_to_do"
                ]
            }
        )
    )
    return parse_gemini_json(response.text)

# =========================================================
# ASK SAFETY QUESTION
# =========================================================
@app.post("/ask_safety")
def ask_safety(req: SafetyRequest):
    prompt = f"""
You are a medical safety assistant for the general public in India.

TASKS:
1. Answer safely.
2. Assign risk level: low, moderate, high.
3. Explain why.
4. Give safe guidance.
5. Say what NOT to do.
6. Say when to see a doctor.
7. Mention a common misconception if relevant.

RULES:
- Do NOT prescribe medicines.
- Do NOT suggest dosage.
- Do NOT override doctors.
- Be conservative and clear.

Respond STRICTLY in this language: {req.lang}
Respond STRICTLY in valid JSON.

Question:
{req.question}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "short_answer": {"type": "string"},
                    "why": {"type": "string"},
                    "what_to_do": {"type": "string"},
                    "what_not_to_do": {"type": "string"},
                    "when_to_see_doctor": {"type": "string"},
                    "common_misconception": {"type": "string"},
                    "risk_level": {
                        "type": "string",
                        "enum": ["low", "moderate", "high"]
                    }
                },
                "required": [
                    "short_answer",
                    "why",
                    "what_to_do",
                    "what_not_to_do",
                    "when_to_see_doctor",
                    "risk_level"
                ]
            }
        )
    )
    return parse_gemini_json(response.text)

# =========================================================
# CHECK IMAGE — MEDICINE STRIP
# =========================================================
@app.post("/check_image")
async def check_image(
    file: UploadFile = File(...),
    lang: str = Form("en")
):
    image_bytes = await file.read()

    text_part = types.Part.from_text(text=f"""
You are a medical safety assistant for the Indian healthcare context.

TASKS:
1. Identify ACTIVE GENERIC / CHEMICAL NAME(S).
2. Classify into ONE category:
   antibiotic, steroid, painkiller, antipyretic,
   antihistamine, antacid, vitamin/supplement,
   common_otc, others, unknown
3. Provide safety warnings.
4. Mention common misuse in India.
5. Give safe advice.
6. Clearly say what NOT to do.

CONFIDENCE RULE:
If <50% sure, set generic_name = "unknown" and category = "unknown".

RULES:
- No dosage.
- No prescriptions.
- Simple language.
- Encourage doctor consultation.

Respond STRICTLY in this language: {lang}
Respond STRICTLY in valid JSON.
""")

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=file.content_type
    )

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[types.Content(parts=[text_part, image_part])],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "generic_name": {"type": "string"},
                    "category": {"type": "string"},
                    "warnings": {"type": "string"},
                    "advice": {"type": "string"}
                },
                "required": ["generic_name", "category", "warnings", "advice"]
            }
        )
    )
    return parse_gemini_json(response.text)

# Run with:
# uvicorn main:app --reload