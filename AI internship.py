
CodeAlpha AI Internship — FastAPI Backend
Tasks: 1 · Language Translation  |  2 · FAQ Chatbot  |  4 · Object Detection
Uses: OpenAI GPT-4o-mini  +  deep-translator  +  YOLOv8n + OpenCV
"""

import os, base64, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import openai
import numpy as np
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("codealpha")

# ── Globals ────────────────────────────────────────────────────────────────────
_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        try:
            from ultralytics import YOLO
            log.info("Loading YOLOv8n …")
            _yolo = YOLO("yolov8n.pt")
            log.info("YOLOv8n ready ✓")
        except Exception as e:
            log.warning(f"YOLOv8 unavailable: {e}")
    return _yolo

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀  CodeAlpha AI API starting …")
    get_yolo()          # pre-warm model so first request is fast
    yield
    log.info("API shut down.")

app = FastAPI(title="CodeAlpha AI API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client — reads OPENAI_API_KEY from environment / .env
_openai_client: Optional[openai.OpenAI] = None

def get_openai() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise HTTPException(
                status_code=503,
                detail="OPENAI_API_KEY is not set. Add it to backend/.env",
            )
        _openai_client = openai.OpenAI(api_key=key)
    return _openai_client

# ── Schemas ────────────────────────────────────────────────────────────────────

SUPPORTED_LANGUAGES: dict[str, str] = {
    "auto": "Auto Detect",
    "en": "English",   "hi": "Hindi",     "te": "Telugu",  "ta": "Tamil",
    "fr": "French",    "de": "German",    "es": "Spanish", "it": "Italian",
    "pt": "Portuguese","ru": "Russian",   "zh-CN": "Chinese (Simplified)",
    "ja": "Japanese",  "ko": "Korean",    "ar": "Arabic",  "nl": "Dutch",
    "pl": "Polish",    "sv": "Swedish",   "tr": "Turkish", "vi": "Vietnamese",
}

class TranslateRequest(BaseModel):
    text:   str = Field(..., max_length=2000)
    source: str = "auto"
    target: str = "en"

class ChatMessage(BaseModel):
    role:    str   # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    topic:    Optional[str] = "artificial intelligence"

# ── Task 1 — Language Translation ──────────────────────────────────────────────

@app.get("/api/languages")
def get_languages():
    return {"languages": SUPPORTED_LANGUAGES}

@app.post("/api/translate")
def translate_text(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Text cannot be empty.")

    # Same language — skip API call
    if req.source == req.target and req.source != "auto":
        return {"translated": text, "source": req.source, "target": req.target}

    src_name = SUPPORTED_LANGUAGES.get(req.source, req.source)
    tgt_name = SUPPORTED_LANGUAGES.get(req.target, req.target)

    # Use GPT-4o-mini for translation — zero setup, no separate translation key
    client = get_openai()
    src_clause = f"from {src_name}" if req.source != "auto" else "from whatever language it is written in"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1500,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a professional translator. "
                        f"Translate the user's text {src_clause} to {tgt_name}. "
                        "Return ONLY the translated text — no explanations, no quotes, no extra words."
                    ),
                },
                {"role": "user", "content": text},
            ],
        )
        translated = resp.choices[0].message.content.strip()
        return {"translated": translated, "source": req.source, "target": req.target}
    except openai.AuthenticationError:
        raise HTTPException(401, "Invalid OPENAI_API_KEY.")
    except openai.RateLimitError:
        raise HTTPException(429, "OpenAI rate limit hit. Try again in a moment.")
    except Exception as e:
        log.error(f"Translation error: {e}")
        raise HTTPException(500, f"Translation failed: {str(e)}")

# ── Task 2 — FAQ Chatbot ───────────────────────────────────────────────────────

CHAT_SYSTEM = """You are a knowledgeable AI teaching assistant for CodeAlpha internship students studying {topic}.

Guidelines:
- Keep answers concise: 2–4 sentences unless detail is genuinely needed.
- Use plain language; give concrete examples where helpful.
- For code questions, use short inline code snippets with backticks.
- Be encouraging — students may be beginners.

Your areas of expertise: machine learning, deep learning, neural networks, NLP, computer vision,
Python, NumPy, TensorFlow, PyTorch, Scikit-learn, YOLO, transformers, model evaluation, AI ethics,
data preprocessing, overfitting, regularisation, GANs, RNNs, LSTMs, attention mechanisms."""

@app.post("/api/chat")
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "Messages list is empty.")

    client = get_openai()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=700,
            temperature=0.7,
            messages=[
                {"role": "system", "content": CHAT_SYSTEM.format(topic=req.topic or "AI")},
                *[{"role": m.role, "content": m.content} for m in req.messages],
            ],
        )
        reply = resp.choices[0].message.content.strip()
        return {"reply": reply}
    except openai.AuthenticationError:
        raise HTTPException(401, "Invalid OPENAI_API_KEY. Check your backend/.env file.")
    except openai.RateLimitError:
        raise HTTPException(429, "OpenAI rate limit reached. Please wait a moment.")
    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(500, f"Chat error: {str(e)}")

# ── Task 4 — Object Detection ─────────────────────────────────────────────────

# 20 visually distinct BGR colours (one per COCO class family)
_PALETTE = [
    (255, 56,  56),  (255,157,151),  (255,112, 31),  (255,178, 29),  (207,210, 49),
    ( 72,249, 10),  (146,204, 23),  ( 61,219,134),  ( 26,147, 52),  (  0,212,187),
    ( 44,153,168),  (  0,194,255),  ( 52, 69,147),  (100,115,255),  (  0, 24,236),
    (132, 56,255),  ( 82,  0,133),  (203, 56,255),  (255,149,200),  (255, 55,199),
]

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    model = get_yolo()
    if model is None:
        raise HTTPException(503, "YOLOv8 not available. Run: pip install ultralytics")

    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "File must be an image (JPG, PNG, WEBP, etc.).")

    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image. Please upload a valid image file.")

    results    = model(img, conf=0.40, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf  = float(box.conf[0])
            cid   = int(box.cls[0])
            label = result.names[cid]
            color = _PALETTE[cid % len(_PALETTE)]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Label pill
            text = f" {label}  {conf:.0%} "
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
            label_y = max(y1 - th - 8, 0)
            cv2.rectangle(img, (x1, label_y), (x1 + tw, label_y + th + 8), color, -1)
            cv2.putText(img, text, (x1, label_y + th + 3), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

            detections.append({
                "label":      label,
                "confidence": round(conf, 4),
                "bbox":       [x1, y1, x2, y2],
                "class_id":   cid,
            })

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    b64    = base64.b64encode(buf).decode()

    log.info(f"Detected {len(detections)} object(s) in '{file.filename}'")
    return {
        "image":      f"data:image/jpeg;base64,{b64}",
        "detections": detections,
        "count":      len(detections),
    }

# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "yolo":   "yolov8n" if _yolo else "loading",
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "tasks":  ["translation", "chatbot", "object-detection"],
    }
