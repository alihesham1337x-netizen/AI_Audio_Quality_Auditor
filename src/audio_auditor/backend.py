import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List

from .pipeline import AudioAuditPipeline


def create_app() -> FastAPI:
    app = FastAPI(title="AI Audio Quality Auditor")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    pipeline = AudioAuditPipeline()
    os.makedirs("./temp_uploads", exist_ok=True)

    @app.post("/audit")
    async def audit_files(
        files: List[UploadFile] = File(...),
        sensitivity: float = Form(0.5),
    ):
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        energy_threshold = max(0.01, min(0.05, 0.04 - sensitivity * 0.03))
        results = []
        for uploaded in files:
            contents = await uploaded.read()
            temp_path = os.path.join("./temp_uploads", uploaded.filename)
            with open(temp_path, "wb") as f:
                f.write(contents)
            results.append(pipeline.audit_file(temp_path, sustained_energy_threshold=energy_threshold))

        return JSONResponse(content=[result.__dict__ for result in results])

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
