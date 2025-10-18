from fastapi import APIRouter, UploadFile, File
from audio_classifier.services.classification_service import classify_audio

router = APIRouter(prefix="/api", tags=["Audio Classification"])

@router.post("/classify")
async def classify_audio_endpoint(file: UploadFile = File(...)):
    """Nhận file âm thanh và trả về kết quả phân loại."""
    audio_bytes = await file.read()
    result = classify_audio(audio_bytes)
    
    # Transform the result to match frontend expectations
    return {
        "status": "success", 
        "result": {
            "label": result.get("final_category", "Unknown"),
            "confidence": result.get("category_scores", {}).get(result.get("final_category"), 0)
        }
    }
