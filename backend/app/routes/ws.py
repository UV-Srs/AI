import json
import base64
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.inference.pipeline import run_integrated_pipeline

router = APIRouter(tags=["Live WebSocket Detection"])

@router.websocket("/ws/detection")
async def detection_websocket_endpoint(websocket: WebSocket):
    """
    Real-time bidirectional WebSocket endpoint for webcam frames.
    Executes multimodal live detection (YOLO26n PPE + OpenCV Face Recognition)
    and returns combined compliance results.
    """
    await websocket.accept()
    print("[Live WebSocket] Client connected for multimodal PPE & Face recognition stream.", flush=True)
    
    try:
        while True:
            # Receive either binary frame or text JSON
            message = await websocket.receive()
            
            camera_id = "CAM-01"
            ppe_conf = 0.08
            face_thresh = None
            imgsz = 416

            if "bytes" in message and message["bytes"]:
                image_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    raw_data = payload.get("image", "")
                    
                    if "," in raw_data:
                        raw_data = raw_data.split(",", 1)[1]
                        
                    image_bytes = base64.b64decode(raw_data)
                    ppe_conf = float(payload.get("conf_threshold", 0.08))
                    class_thresh = payload.get("class_thresholds", None)
                    if "face_match_threshold" in payload:
                        face_thresh = float(payload["face_match_threshold"])
                    camera_id = payload.get("camera_id", "CAM-01")
                    imgsz = int(payload.get("imgsz", 416))
                except Exception as e:
                    await websocket.send_json({
                        "success": False,
                        "error": f"Invalid frame payload: {str(e)}"
                    })
                    continue
            else:
                continue

            # Run integrated pipeline in worker thread to avoid blocking asyncio event loop
            pipeline_result = await asyncio.to_thread(
                run_integrated_pipeline,
                image_bytes,
                ppe_conf_threshold=ppe_conf,
                face_match_threshold=face_thresh,
                class_thresholds=class_thresh,
                camera_id=camera_id,
                imgsz=imgsz
            )

            # Send full multimodal result back to browser
            await websocket.send_json(pipeline_result)

    except WebSocketDisconnect:
        print("[Live WebSocket] Client disconnected gracefully.", flush=True)
    except Exception as e:
        print(f"[Live WebSocket] Streaming error: {e}", flush=True)
        try:
            await websocket.close()
        except Exception:
            pass
