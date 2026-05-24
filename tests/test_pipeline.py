import os
import pytest

from src.audio_auditor.pipeline import AudioAuditPipeline


def test_audit_pipeline_no_file():
    pipeline = AudioAuditPipeline()
    with pytest.raises(FileNotFoundError):
        pipeline.audit_file("nonexistent.wav")


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
