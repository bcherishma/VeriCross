from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models.schemas import EvaluationRequest, EvaluationResponse
from app.utils.detector import HallucinationDetector
import json

router = APIRouter()
detector = HallucinationDetector()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_summary(request: EvaluationRequest):
    result = detector.evaluate(request.source_text, request.summary_text, request.source_lang, request.summary_lang)
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