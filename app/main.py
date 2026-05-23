from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.upload import router as upload_router
from app.routes.predict import router as predict_router
from app.routes.explain import router as explain_router
from app.routes.report_predict import router as report_router

app = FastAPI(
    title="PulseKin Healthcare AI Backend",
    version="2.0.0"
)

# IMPORTANT: In production, replace "*" with your actual frontend domain
# e.g. ["https://pulsekin.vercel.app"] or ["http://localhost:3000"]
ALLOWED_ORIGINS = ["*"]  # TODO: restrict before going live

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,   # Must be False when allow_origins=["*"]
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(upload_router)
app.include_router(predict_router)
app.include_router(explain_router)
app.include_router(report_router)


@app.get("/")
def home():
    return {"message": "PulseKin Healthcare AI Backend Running", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}