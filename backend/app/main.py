from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes import api, ws
from backend.app.services.model_manager import model_manager

app = FastAPI(
    title="UNIFIED PPE & 3D FACE RECOGNITION SYSTEM",
    description="Local real-time PPE inspection & facial recognition powered by YOLO26n, YuNet, MediaPipe 3D Mesh, and dlib ResNet-34",
    version="2.0.0"
)

# Enable CORS for local React/Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5174",
        "http://127.0.0.1:5174"
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(api.router)
app.include_router(ws.router)

@app.on_event("startup")
async def startup_event():
    status = model_manager.get_status()
    print("=" * 70)
    print("UNIFIED PPE DETECTION & 3D FACE RECOGNITION BACKEND READY")
    print(f"Device:       {status['device'].upper()}")
    print(f"Active Model: {status['active_model_name']}")
    print(f"Model Loaded: {status['model_loaded']}")
    print(f"Classes:      {status['classes']}")
    print("=" * 70)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8001, reload=False)

