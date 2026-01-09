from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models.schemas import EvaluationRequest, EvaluationResponse
from app.utils.detector import HallucinationDetector
import json

router = APIRouter()
detector = HallucinationDetector()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_summary(request: EvaluationRequest):
    # Auto-detect languages if not provided or if "auto" is specified
    source_lang = request.source_lang
    summary_lang = request.summary_lang
    
    if not source_lang or source_lang.lower() == "auto":
        source_lang = detector.detect_language(request.source_text)
    
    if not summary_lang or summary_lang.lower() == "auto":
        summary_lang = detector.detect_language(request.summary_text)
    
    result = detector.evaluate(request.source_text, request.summary_text, source_lang, summary_lang)
    
    # Add detected languages to the response
    result["detected_source_lang"] = source_lang
    result["detected_summary_lang"] = summary_lang
    
    return EvaluationResponse(**result)

@router.websocket("/ws/evaluate")
async def websocket_evaluate(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            request = EvaluationRequest(**json.loads(data))
            result = detector.evaluate(request.source_text, request.summary_text, request.source_lang, request.summary_lang)
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass