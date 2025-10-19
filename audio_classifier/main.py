from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from audio_classifier.api.routes import router
from audio_classifier.core.ml_models.predictor import YamnetPredictor
import os
from contextlib import asynccontextmanager


# -------------------------------
# ⚙️ Lifespan handler
# -------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading YAMNet + ANN classifier on startup...")
    _ = YamnetPredictor()  # preload model
    yield
    print("🛑 Shutting down app...")


# -------------------------------
# 🌐 Create app
# -------------------------------
app = FastAPI(title="Audio Classifier API", version="1.0.0", lifespan=lifespan)

# Gắn router API
app.include_router(router)


# -------------------------------
# 📁 Static & frontend
# -------------------------------
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
def serve_frontend():
    """Trả về giao diện test upload file âm thanh."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Audio Classification API is running, nhưng chưa có file index.html!"}


# -------------------------------
# 🏁 Entry point
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("audio_classifier.main:app", host="0.0.0.0", port=8000, reload=True)
