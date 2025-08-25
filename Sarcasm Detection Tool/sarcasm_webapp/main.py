import os, importlib
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# --- FastAPI app ---
app = FastAPI(title="Sarcasm Detector (Audio)")

# main.py (top-level config)
MODEL_PATH = str(BASE_DIR / "best_cmgat_model.pth")
HIDDEN_DIM = 128          # <-- was 256; set to 128 to match your checkpoint
NUM_LAYERS = 3            # <-- your logs show layers.0, layers.1, layers.2 => 3 layers


# --- Static mounting ---
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Fallback HTML UI ---
FALLBACK_HTML = """
<!doctype html><meta charset="utf-8">
<title>Sarcasm Detector</title>
<h1>Sarcasm Detector</h1>
<p>Upload an audio file and get sarcasm prediction.</p>
<input id="file" type="file" accept="audio/*" />
<button id="btn" disabled>Predict</button>
<pre id="out"></pre>
<script>
const f=document.getElementById('file'),b=document.getElementById('btn'),o=document.getElementById('out');
f.addEventListener('change',()=>b.disabled=!f.files.length);
b.addEventListener('click',async()=>{
  const fd=new FormData(); fd.append('audio', f.files[0]); o.textContent='Processing...';
  const res=await fetch('/predict',{method:'POST',body:fd});
  const data=await res.json(); o.textContent=JSON.stringify(data,null,2);
});
</script>
"""

# --- Root route ---
@app.get("/", response_class=HTMLResponse)
def home():
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return idx.read_text(encoding="utf-8")
    return FALLBACK_HTML

# --- Lazy import inference ---
def _get_predict():
    m = importlib.import_module("app.inference")
    return m.predict_from_file

# --- Prediction endpoint ---
@app.post("/predict")
async def predict(audio: UploadFile = File(...)):
    with NamedTemporaryFile(delete=False, suffix=Path(audio.filename).suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        predict_from_file = _get_predict()
        result = predict_from_file(tmp_path, MODEL_PATH, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

# --- Entry point ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
