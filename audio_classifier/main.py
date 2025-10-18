from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from audio_classifier.api.routes import router
import os

# Tạo ứng dụng FastAPI
app = FastAPI(title="Audio Classifier API", version="1.0.0")

# Gắn router API
app.include_router(router)

# Đường dẫn tới thư mục chứa index.html
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Đảm bảo thư mục static tồn tại (để không lỗi khi mount)
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Mount static directory (chứa file index.html)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Route gốc trả về giao diện web test
@app.get("/", response_class=FileResponse)
def serve_frontend():
    """Trả về giao diện test upload file âm thanh."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Audio Classification API is running, nhưng chưa có file index.html!"}

# Entry point để chạy app trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("audio_classifier.main:app", host="0.0.0.0", port=8000, reload=True)
