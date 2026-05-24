import os
from pathlib import Path

import streamlit as st

from src.audio_auditor.pipeline import AudioAuditPipeline


def main():
    st.title("AI Audio Quality Auditor")
    st.write("Upload your `.mp3` or `.wav` call recordings for batch audit.")

    uploaded_files = st.file_uploader("Choose audio files", type=["mp3", "wav"], accept_multiple_files=True)
    if uploaded_files:
        pipeline = AudioAuditPipeline()
        results = []
        upload_dir = Path("./temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        for uploaded in uploaded_files:
            target_path = upload_dir / uploaded.name
            with open(target_path, "wb") as f:
                f.write(uploaded.getbuffer())
            results.append(pipeline.audit_file(str(target_path)))

        st.header("Audit Results")
        for result in results:
            st.subheader(result.file_path)
            st.write(f"Decision: **{result.overall_decision}**")
            st.write(f"Total severity: {result.total_severity:.2f}")
            st.table([{
                "Start": event["start"],
                "End": event["end"],
                "Label": event["label"],
                "Severity": f"{event['severity']:.1f}",
                "Confidence": f"{event['confidence']:.2f}",
            } for event in result.events])

        if st.button("Export JSON report"):
            out_path = "audit_report.json"
            pipeline.export_json(results, out_path)
            st.success(f"Report saved to {out_path}")


if __name__ == "__main__":
    main()
