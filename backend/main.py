#pip install "google-genai>=0.1.0" fastapi uvicorn python-multipart pydantic
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


# --- CONFIGURATION ---
# Replace with your actual key
API_KEY = "AIzaSyCNLdpfDCRQMm8CYL61Z7tJztnP2nnfl8w"

client = genai.Client(api_key=API_KEY)

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

class SafetyRequest(BaseModel):
    question: str

# Helper to clean JSON response (removes ```json ... ``` if present)
def parse_gemini_json(text_response: str):
    clean_text = text_response.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": text_response}

# ---------------------------
# CHECK TEXT (MISINFO)
# ---------------------------
@app.post("/check_text")
def check_text(req: TextRequest):
    try:
        prompt = f"""
        You are a medical safety assistant.
        Classify this claim as: misinformation or reliable.
        
        Claim: {req.text}
        """
        
        # We enforce JSON response schema for reliability
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "string"}
                    }
                }
            )
        )
        
        return parse_gemini_json(response.text)

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# ASK SAFETY QUESTION
# ---------------------------
@app.post("/ask_safety")
def ask_safety(req: SafetyRequest):
    try:
        prompt = f"""
        You are a medical safety assistant. Answer safely. Do NOT prescribe drugs.
        Question: {req.question}
        """
        
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"}
                    }
                }
            )
        )

        return parse_gemini_json(response.text)

    except Exception as e:
        return {"error": str(e)}


# ---------------------------
# CHECK IMAGE
# ---------------------------
@app.post("/check_image")
async def check_image(file: UploadFile = File(...)):
    try:
        # 1. Read the bytes
        image_bytes = await file.read()
        
        # 2. Prepare the prompt part
        text_part = types.Part.from_text(text="""
        Look at this medicine strip.
        Classify ONLY as: antibiotic, steroid, common_otc, or unknown.
        Provide a warning if necessary.
        """)

        # 3. Prepare the image part (CRITICAL FIX)
        # We must tell Gemini what kind of file this is (jpeg, png, etc.)
        image_part = types.Part.from_bytes(
            data=image_bytes, 
            mime_type=file.content_type
        )

        # 4. Generate with JSON config
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[types.Content(parts=[text_part, image_part])],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "warning": {"type": "string"}
                    }
                }
            )
        )

        return parse_gemini_json(response.text)

    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn main:app --reload