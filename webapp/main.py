# webapp/main.py
import os
import uuid
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pipeline import run_pipeline

app = FastAPI(title="PDF AI Enhancer")
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else os.getcwd()
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: serve static assets if you add them
if os.path.isdir(os.path.join(BASE_DIR, "webapp", "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "webapp", "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    # Minimal, clean UI that auto-downloads when ready and shows a spinner
    return """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>PDF AI Enhancer</title>
      <style>
        body { font-family: Inter, Arial, sans-serif; background: #f6f8fb; display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }
        .card { background:white; padding:28px; border-radius:12px; box-shadow:0 6px 24px rgba(20,30,60,0.08); width:520px; text-align:center; }
        h1 { margin:0 0 8px 0; font-size:20px; color:#0f1724; }
        p { color:#475569; margin:0 0 18px 0; }
        input[type=file] { display:block; margin:18px auto; }
        button { background:#0ea5e9; color:white; border:none; padding:10px 18px; border-radius:8px; cursor:pointer; font-weight:600; }
        button:disabled { opacity:0.6; cursor:not-allowed; }
        .status { margin-top:16px; color:#334155; }
        .spinner { border:4px solid #e6eef6; border-top:4px solid #0ea5e9; border-radius:50%; width:28px; height:28px; animation:spin 1s linear infinite; display:inline-block; vertical-align:middle; margin-right:8px; }
        @keyframes spin { to { transform: rotate(360deg); } }
      </style>
    </head>
    <body>
      <div class="card">
        <h1>PDF AI Enhancer</h1>
        <p>Upload a scanned PDF. The enhanced PDF will download automatically when ready.</p>
        <form id="uploadForm">
          <input type="file" name="file" accept="application/pdf" required />
          <button type="submit">Upload & Enhance</button>
        </form>
        <div class="status" id="status"></div>
      </div>

      <script>
        const form = document.getElementById('uploadForm');
        const status = document.getElementById('status');

        form.onsubmit = async (e) => {
          e.preventDefault();
          const fileInput = form.querySelector('input[type=file]');
          if (!fileInput.files.length) return;
          const file = fileInput.files[0];
          status.innerHTML = '<span class="spinner"></span>Uploading and processing...';
          const fd = new FormData();
          fd.append('file', file, file.name);

          try {
            const res = await fetch('/upload', { method: 'POST', body: fd });
            const data = await res.json();
            if (data.download_url) {
              // Trigger automatic download
              const a = document.createElement('a');
              a.href = data.download_url;
              a.download = '';
              document.body.appendChild(a);
              a.click();
              a.remove();
              status.innerText = 'Download started. Thank you.';
            } else {
              status.innerText = 'Error: ' + (data.error || JSON.stringify(data));
            }
          } catch (err) {
            status.innerText = 'Upload failed: ' + err;
          }
        };
      </script>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "Please upload a PDF file"}

    input_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.pdf")
    with open(input_path, "wb") as f:
        f.write(await file.read())

    output_filename = f"{uuid.uuid4()}_enhanced.pdf"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        # Synchronous processing; for long jobs consider BackgroundTasks or a job queue
        run_pipeline(input_path, output_path)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    if not os.path.exists(output_path):
        return {"error": f"Output file not found at {output_path}"}

    # Return a direct download URL (relative)
    return {"download_url": f"/download/{output_filename}"}


@app.get("/download/{filename}")
async def download_pdf(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return {"error": "File not found"}
    return FileResponse(path, media_type="application/pdf", filename=filename)
