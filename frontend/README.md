# AI Audio Quality Auditor Frontend

This directory contains a Vite + React UI for the FastAPI backend.

## Start locally

```bash
cd frontend
npm install
npm run dev
```

The app will proxy API requests to `http://localhost:8000` via `/api/audit`.

## Features

- Drag-and-drop / file selection support
- Bulk audio upload
- Audit result table display
- Severity and decision summaries
