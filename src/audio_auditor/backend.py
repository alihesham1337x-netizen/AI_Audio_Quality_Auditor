import os
import tempfile
from pathlib import Path
from typing import List

from .pipeline import AudioAuditPipeline


def create_app():
    """Create the FastAPI app. Only call this when fastapi is installed."""
    try:
        from fastapi import FastAPI, UploadFile, File, Form, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
    except ImportError:
        raise ImportError(
            "fastapi is required to run the API backend. "
            "Install it with: pip install fastapi uvicorn[standard]"
        )

    app = FastAPI(title="AI Audio Quality Auditor")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    pipeline = AudioAuditPipeline()
    uploads_dir = Path("./temp_uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)

    @app.post("/audit")
    async def audit_files(
        files: List[UploadFile] = File(default=[]),
        sensitivity: float = Form(0.5),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        energy_threshold = max(0.01, min(0.05, 0.04 - sensitivity * 0.03))
        results = []
        for uploaded in files:
            contents = await uploaded.read()
            suffix = Path(uploaded.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(dir=uploads_dir, delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                temp_path = tmp.name
            results.append(pipeline.audit_file(temp_path, sustained_energy_threshold=energy_threshold))

        return JSONResponse(content=[result.__dict__ for result in results])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
