# AI Audio Quality Auditor

Enterprise-grade audio auditing prototype for VA cold-calling agencies.
This repository contains a modular Python pipeline for detecting sudden environmental background disturbances in call recordings while ignoring normal speech.

## Features

- Bulk `.mp3` / `.wav` ingest
- Audio normalization and resampling to 16 kHz
- Voice activity detection (Silero / WebRTC) for speech/non-speech masking
- Optional source separation using Demucs / Spleeter
- Transient/anomaly detection with spectral flux and onset strength
- Environmental sound classification using YAMNet/PANNs
- Severity scoring and audit decisions: PASS / WARNING / FAIL
- JSON / CSV report generation
- Streamlit MVP UI

## Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the Streamlit MVP

```bash
python app.py
```

## Run the FastAPI backend

```bash
python fastapi_app.py
```

Then connect a frontend or send multipart uploads to `/audit`.

The React UI also supports exporting audit results as JSON or CSV.

## Run tests

```bash
python -m pytest
```

## React frontend

A lightweight React UI is available under `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

The React app proxies `/api/audit` to the FastAPI backend on `http://localhost:8000`.

## Project Layout

- `src/audio_auditor/` — core inference pipeline and utilities
- `app.py` — Streamlit prototype UI
- `tests/` — basic test scaffolding
