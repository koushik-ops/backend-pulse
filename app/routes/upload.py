import os
import uuid
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_MB = 10


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a file, returns a temp reference. File is deleted after processing."""
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Use a random temp filename — never store the original filename on disk
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"pulsekin_{uuid.uuid4().hex}{ext}"
    )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit"
        )

    with open(temp_path, "wb") as f:
        f.write(contents)

    return {
        "temp_path": temp_path,
        "message": "File uploaded. Will be deleted after analysis."
    }