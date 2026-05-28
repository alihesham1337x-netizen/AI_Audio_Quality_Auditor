import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from src.audio_auditor.backend import create_app


def test_health_endpoint():
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_audit_no_files():
    client = TestClient(create_app())
    response = client.post("/audit", data={"sensitivity": "0.5"})

    assert response.status_code == 400
    assert "No files uploaded" in response.text


def test_audit_with_small_wav(tmp_path):
    client = TestClient(create_app())
    fs = 16000
    tone = np.zeros(fs, dtype=np.float32)
    tone[: fs // 4] = 0.1 * np.sin(2 * np.pi * 440 * np.arange(fs // 4) / fs)
    file_path = tmp_path / "tone.wav"
    sf.write(str(file_path), tone, fs)

    with open(file_path, "rb") as audio_file:
        response = client.post(
            "/audit",
            files={"files": ("tone.wav", audio_file, "audio/wav")},
            data={"sensitivity": "0.5"},
        )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    audit_result = data[0]
    assert "overall_decision" in audit_result
    assert "file_path" in audit_result
    assert audit_result["file_path"].endswith("tone.wav") or audit_result["file_path"].endswith(".wav")
