from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json, os, time, uuid
from datetime import datetime
from engine import analyze_documents

app = FastAPI(title="Semantic Plagiarism Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Simple file-based history store
HISTORY_FILE = "history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    with open(HISTORY_FILE) as f:
        try: return json.load(f)
        except: return []

def save_history(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:50]  # keep last 50
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", encoding="utf-8") as f:
        return f.read()


@app.post("/analyze")
async def analyze(
    source_text: str = Form(default=""),
    suspect_text: str = Form(default=""),
    source_file: UploadFile = File(default=None),
    suspect_file: UploadFile = File(default=None),
):
    # Read from file if uploaded
    if source_file and source_file.filename:
        content = await source_file.read()
        source_text = content.decode("utf-8", errors="ignore")
    if suspect_file and suspect_file.filename:
        content = await suspect_file.read()
        suspect_text = content.decode("utf-8", errors="ignore")

    if not source_text.strip() or not suspect_text.strip():
        raise HTTPException(400, "Both documents are required.")

    if len(source_text) > 10000 or len(suspect_text) > 10000:
        raise HTTPException(400, "Documents too long (max 10,000 chars each).")

    start = time.time()
    result = analyze_documents(source_text.strip(), suspect_text.strip())
    result["time_taken"] = round(time.time() - start, 2)

    if "error" not in result:
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now().strftime("%d %b %Y, %H:%M"),
            "score": result["score"],
            "verdict": result["verdict"],
            "verdict_level": result["verdict_level"],
            "source_preview": source_text[:80] + ("..." if len(source_text) > 80 else ""),
            "suspect_preview": suspect_text[:80] + ("..." if len(suspect_text) > 80 else ""),
        }
        save_history(entry)

    return JSONResponse(result)


@app.get("/history")
async def get_history():
    return JSONResponse(load_history())


@app.delete("/history")
async def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    return {"status": "cleared"}
