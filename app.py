import io
import json
from pathlib import Path

import streamlit as st
import requests
from streamlit_lottie import st_lottie

from src.audio_auditor.pipeline import AudioAuditPipeline

st.set_page_config(page_title="Audio Quality Auditor", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="collapsed")

for k, v in [("dark_mode", True), ("sensitivity", 50), ("results", None)]:
    if k not in st.session_state:
        st.session_state[k] = v

def load_lottie(url):
    try:
        r = requests.get(url, timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

LOTTIE = load_lottie("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

def inject_css(dark):
    if dark:
        card = "rgba(10,14,30,0.82)";  bdr = "rgba(255,255,255,0.10)"
        t1   = "#e2e8f0";  t2 = "#94a3b8"
        tr   = "rgba(10,14,30,0.50)";  ta = "rgba(20,28,55,0.50)"
        th   = "rgba(255,255,255,0.07)"; mbg = "rgba(255,255,255,0.05)"
        pc   = "#22c55e"; wc = "#f59e0b"; fc = "#ef4444"
        base_bg = "#07091a"
    else:
        card = "rgba(255,255,255,0.82)"; bdr = "rgba(0,0,0,0.10)"
        t1   = "#1e293b"; t2 = "#475569"
        tr   = "rgba(255,255,255,0.75)"; ta = "rgba(240,244,255,0.75)"
        th   = "rgba(0,0,0,0.06)";       mbg = "rgba(0,0,0,0.04)"
        pc   = "#16a34a"; wc = "#d97706"; fc = "#dc2626"
        base_bg = "#f0f4ff"

    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box}}

/* ── RGB / rainbow background ── */
@keyframes rgbBg {{
  0%   {{ background-position: 0% 50%; }}
  50%  {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}
@keyframes rgbBorder {{
  0%   {{ border-color: rgba(99,102,241,0.55); box-shadow: 0 0 8px rgba(99,102,241,0.3); }}
  14%  {{ border-color: rgba(139,92,246,0.55); box-shadow: 0 0 8px rgba(139,92,246,0.3); }}
  28%  {{ border-color: rgba(236,72,153,0.55); box-shadow: 0 0 8px rgba(236,72,153,0.3); }}
  42%  {{ border-color: rgba(239,68,68,0.55);  box-shadow: 0 0 8px rgba(239,68,68,0.3); }}
  57%  {{ border-color: rgba(234,179,8,0.55);  box-shadow: 0 0 8px rgba(234,179,8,0.3); }}
  71%  {{ border-color: rgba(34,197,94,0.55);  box-shadow: 0 0 8px rgba(34,197,94,0.3); }}
  85%  {{ border-color: rgba(6,182,212,0.55);  box-shadow: 0 0 8px rgba(6,182,212,0.3); }}
  100% {{ border-color: rgba(99,102,241,0.55); box-shadow: 0 0 8px rgba(99,102,241,0.3); }}
}}
@keyframes rgbText {{
  0%   {{ color: #818cf8; }}
  14%  {{ color: #a78bfa; }}
  28%  {{ color: #f472b6; }}
  42%  {{ color: #f87171; }}
  57%  {{ color: #fbbf24; }}
  71%  {{ color: #4ade80; }}
  85%  {{ color: #22d3ee; }}
  100% {{ color: #818cf8; }}
}}
@keyframes rgbGlow {{
  0%   {{ box-shadow: 0 4px 20px rgba(99,102,241,0.5); }}
  14%  {{ box-shadow: 0 4px 20px rgba(139,92,246,0.5); }}
  28%  {{ box-shadow: 0 4px 20px rgba(236,72,153,0.5); }}
  42%  {{ box-shadow: 0 4px 20px rgba(239,68,68,0.5); }}
  57%  {{ box-shadow: 0 4px 20px rgba(234,179,8,0.5); }}
  71%  {{ box-shadow: 0 4px 20px rgba(34,197,94,0.5); }}
  85%  {{ box-shadow: 0 4px 20px rgba(6,182,212,0.5); }}
  100% {{ box-shadow: 0 4px 20px rgba(99,102,241,0.5); }}
}}
@keyframes rgbBtn {{
  0%   {{ background-position: 0% 50%; }}
  50%  {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}

.stApp, .main, .block-container {{
  font-family: 'Inter', sans-serif !important;
  color: {t1} !important;
}}
.stApp {{
  background: linear-gradient(
    -45deg,
    {'#07091a, #0d0a2e, #0a1a0d, #1a0a0a, #0a0d1a, #1a0a14, #07091a' if dark else
     '#f0f4ff, #f5f0ff, #fff0f5, #f0fff4, #f0f8ff, #fff5f0, #f0f4ff'}
  ) !important;
  background-size: 400% 400% !important;
  animation: rgbBg 18s ease infinite !important;
}}
.block-container {{
  padding: .8rem 1.6rem 2rem !important;
  max-width: 1300px !important;
  background: transparent !important;
}}

/* ── header ── */
.aq-hdr{{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0 .2rem}}
.aq-brand{{display:flex;align-items:center;gap:.6rem}}
.aq-logo{{
  width:36px;height:36px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;font-size:1.2rem;
  background:linear-gradient(-45deg,#6366f1,#8b5cf6,#ec4899,#ef4444,#eab308,#22c55e,#06b6d4,#6366f1);
  background-size:400% 400%;
  animation:rgbBtn 6s ease infinite, rgbGlow 6s ease infinite;
}}
.aq-title{{
  font-size:1.15rem;font-weight:700;color:{t1};line-height:1;
  animation:rgbText 8s ease infinite;
}}
.aq-sub{{font-size:.72rem;color:{t2};margin-top:1px}}

/* ── divider with RGB glow ── */
hr{{
  border: none !important;
  height: 1px !important;
  margin: .6rem 0 !important;
  background: linear-gradient(90deg,transparent,currentColor,transparent);
  animation: rgbText 8s ease infinite;
  opacity: 0.4;
}}

/* ── buttons ── */
.stButton>button, .stDownloadButton>button {{
  background: linear-gradient(-45deg,#6366f1,#8b5cf6,#ec4899,#ef4444,#eab308,#22c55e,#06b6d4,#6366f1) !important;
  background-size: 400% 400% !important;
  animation: rgbBtn 6s ease infinite !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  padding: .35rem 1rem !important;
  font-size: .8rem !important;
  transition: transform .18s !important;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
  transform: translateY(-1px) !important;
}}

/* ── file uploader ── */
.stFileUploader {{
  background: {card} !important;
  border-radius: 10px !important;
  padding: .5rem !important;
  animation: rgbBorder 8s ease infinite;
}}

/* ── slider thumb ── */
.stSlider [data-baseweb="slider"] {{ padding:.3rem 0 !important }}
[data-testid="stSlider"] [role="slider"] {{
  animation: rgbGlow 6s ease infinite;
}}

/* ── expander ── */
.streamlit-expanderHeader {{
  background: {card} !important;
  border-radius: 8px !important;
  font-size: .82rem !important;
  font-weight: 600 !important;
  color: {t1} !important;
  padding: .4rem .7rem !important;
  animation: rgbBorder 8s ease infinite;
}}
.streamlit-expanderContent {{
  background: {card} !important;
  border-top: none !important;
  border-radius: 0 0 8px 8px !important;
  padding: .6rem .7rem !important;
  border-left: 1px solid;
  border-right: 1px solid;
  border-bottom: 1px solid;
  animation: rgbBorder 8s ease infinite;
}}

/* ── decision badges ── */
.b-pass{{background:{pc}18;color:{pc};border:1px solid {pc}40;border-radius:5px;padding:1px 8px;font-weight:700;font-size:.75rem;white-space:nowrap}}
.b-warn{{background:{wc}18;color:{wc};border:1px solid {wc}40;border-radius:5px;padding:1px 8px;font-weight:700;font-size:.75rem;white-space:nowrap}}
.b-fail{{background:{fc}18;color:{fc};border:1px solid {fc}40;border-radius:5px;padding:1px 8px;font-weight:700;font-size:.75rem;white-space:nowrap}}

/* ── results table ── */
.aq-tbl{{width:100%;border-collapse:collapse;font-size:.8rem;border-radius:8px;overflow:hidden;margin-top:.4rem}}
.aq-tbl thead tr{{background:{th}}}
.aq-tbl th{{padding:5px 10px;text-align:left;font-weight:600;color:{t2};font-size:.68rem;text-transform:uppercase;letter-spacing:.04em}}
.aq-tbl tbody tr:nth-child(odd){{background:{tr}}}
.aq-tbl tbody tr:nth-child(even){{background:{ta}}}
.aq-tbl td{{padding:5px 10px;color:{t1};border-bottom:1px solid {bdr}}}
.aq-tbl tbody tr:last-child td{{border-bottom:none}}
.aq-tbl tbody tr:hover{{background:{mbg}!important}}

/* ── severity pills ── */
.sc{{background:#ef444415;color:#ef4444;border:1px solid #ef444435;border-radius:4px;padding:1px 6px;font-size:.68rem;font-weight:600}}
.sh{{background:#f9731615;color:#f97316;border:1px solid #f9731635;border-radius:4px;padding:1px 6px;font-size:.68rem;font-weight:600}}
.sm{{background:#eab30815;color:#eab308;border:1px solid #eab30835;border-radius:4px;padding:1px 6px;font-size:.68rem;font-weight:600}}
.sl{{background:#22c55e15;color:#22c55e;border:1px solid #22c55e35;border-radius:4px;padding:1px 6px;font-size:.68rem;font-weight:600}}

/* ── summary row ── */
.aq-sum{{display:flex;gap:1rem;flex-wrap:wrap;margin:.4rem 0 .6rem;font-size:.78rem}}
.aq-sum-item{{display:flex;align-items:center;gap:.3rem;color:{t2}}}
.aq-sum-val{{font-weight:700;color:{t1}}}

/* ── progress bar ── */
.stProgress>div>div {{
  background: linear-gradient(-45deg,#6366f1,#8b5cf6,#ec4899,#22c55e,#06b6d4,#6366f1) !important;
  background-size: 400% 400% !important;
  animation: rgbBtn 3s ease infinite !important;
  border-radius: 3px !important;
}}

/* ── misc ── */
audio{{border-radius:6px!important;width:100%!important;height:32px!important}}
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-thumb{{
  border-radius:2px;
  background:linear-gradient(180deg,#6366f1,#ec4899,#22c55e);
  animation:rgbBtn 4s ease infinite;
}}
#MainMenu,footer,header{{visibility:hidden}}
</style>""", unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='aq-hdr'>
    <div class='aq-brand'>
        <div class='aq-logo'>🎙️</div>
        <div>
            <div class='aq-title'>AI Audio Quality Auditor</div>
            <div class='aq-sub'>Background noise detection for call-centre recordings</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

theme_col = st.columns([8, 1])[1]
with theme_col:
    if st.button("☀️" if st.session_state.dark_mode else "🌙", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.divider()

# ── Controls row ──────────────────────────────────────────────────────────────
ctl_left, ctl_mid, ctl_right = st.columns([3, 1, 1], gap="medium")

with ctl_left:
    uploaded_files = st.file_uploader(
        "Upload MP3 / WAV files",
        type=["mp3", "wav"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        names = "  ·  ".join(
            f"{f.name[:30]}{'…' if len(f.name)>30 else ''} ({len(f.getvalue())//1024}KB)"
            for f in uploaded_files
        )
        st.markdown(
            f"<div style='font-size:.72rem;opacity:.6;margin-top:.2rem;'>{names}</div>",
            unsafe_allow_html=True,
        )

with ctl_mid:
    st.markdown("<div style='font-size:.72rem;opacity:.6;margin-bottom:.2rem;'>Sensitivity</div>",
                unsafe_allow_html=True)
    sensitivity = st.slider("sens", 0, 100, st.session_state.sensitivity,
                            label_visibility="collapsed")
    st.session_state.sensitivity = sensitivity

with ctl_right:
    st.markdown("<div style='height:1.55rem'></div>", unsafe_allow_html=True)
    audit_clicked = st.button("🚀 Run Audit", type="primary",
                              use_container_width=True, disabled=not uploaded_files)

# ── Audit processing ──────────────────────────────────────────────────────────
if audit_clicked and uploaded_files:
    st.session_state.results = None
    pipeline   = AudioAuditPipeline()
    upload_dir = Path("./temp_uploads")
    upload_dir.mkdir(exist_ok=True)

    prog = st.progress(0)
    stxt = st.empty()

    t     = sensitivity / 100.0
    z_thr = max(1.8, 3.2 - t * 1.4)
    e_thr = max(0.010, 0.030 - t * 0.020)
    f_min = max(0.10,  0.22  - t * 0.12)

    results = []
    for idx, uf in enumerate(uploaded_files):
        stxt.markdown(
            f"<div style='font-size:.75rem;opacity:.65;'>⚙️ Analysing "
            f"<strong>{uf.name}</strong> ({idx+1}/{len(uploaded_files)})</div>",
            unsafe_allow_html=True,
        )
        target = upload_dir / uf.name
        with open(target, "wb") as fh:
            fh.write(uf.getbuffer())
        results.append(pipeline.audit_file(
            str(target),
            sustained_energy_threshold=e_thr,
            transient_z_threshold=z_thr,
            transient_flatness_min=f_min,
        ))
        prog.progress((idx + 1) / len(uploaded_files))

    st.session_state.results = results
    stxt.markdown(
        "<div style='font-size:.75rem;color:#22c55e;'>✅ Done</div>",
        unsafe_allow_html=True,
    )

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    st.divider()

    # ── compact summary line ──────────────────────────────────────────────────
    total_ev = sum(len(r.events) for r in results)
    n_fail   = sum(1 for r in results if r.overall_decision == "FAIL")
    n_warn   = sum(1 for r in results if r.overall_decision == "WARNING")
    n_pass   = sum(1 for r in results if r.overall_decision == "PASS")
    st.markdown(
        f"<div class='aq-sum'>"
        f"<span class='aq-sum-item'>Files <span class='aq-sum-val'>{len(results)}</span></span>"
        f"<span class='aq-sum-item'>Events <span class='aq-sum-val'>{total_ev}</span></span>"
        f"<span class='aq-sum-item'><span class='b-fail'>⛔ {n_fail} FAIL</span></span>"
        f"<span class='aq-sum-item'><span class='b-warn'>⚠️ {n_warn} WARN</span></span>"
        f"<span class='aq-sum-item'><span class='b-pass'>✅ {n_pass} PASS</span></span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    BADGE = {"PASS": "b-pass", "WARNING": "b-warn", "FAIL": "b-fail"}
    ICON  = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "⛔"}

    for result in results:
        fname  = Path(result.file_path).name
        dec    = result.overall_decision
        icon   = ICON.get(dec, "❓")
        bcls   = BADGE.get(dec, "b-warn")
        dur    = result.metadata.get("duration_sec", 0)
        n_ev   = len(result.events)

        # expander title: icon + filename + badge inline
        exp_title = f"{icon}  {fname}"
        with st.expander(exp_title, expanded=(dec != "PASS")):

            # ── one-line file info ────────────────────────────────────────
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:.8rem;margin-bottom:.5rem;'>"
                f"<span class='{bcls}'>{icon} {dec}</span>"
                f"<span style='font-size:.75rem;opacity:.6;'>{dur:.1f}s</span>"
                f"<span style='font-size:.75rem;opacity:.6;'>{n_ev} event{'s' if n_ev!=1 else ''}</span>"
                f"<span style='font-size:.75rem;opacity:.6;'>sev {result.total_severity:.1f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── single playback bar ───────────────────────────────────────
            matched = next((uf for uf in (uploaded_files or []) if uf.name == fname), None)
            if matched:
                fmt = "audio/wav" if fname.lower().endswith(".wav") else "audio/mpeg"
                st.audio(matched, format=fmt)

            # ── events table or clean message ─────────────────────────────
            if not result.events:
                st.markdown(
                    "<div style='font-size:.78rem;color:#22c55e;padding:.4rem 0;'>"
                    "✅ No violations detected</div>",
                    unsafe_allow_html=True,
                )
                continue

            rows = ""
            for ev in result.events:
                s = ev["severity"]
                if s >= 8:
                    pill = f"<span class='sc'>🔴 Critical</span>"
                elif s >= 6:
                    pill = f"<span class='sh'>🟠 High</span>"
                elif s >= 4.5:
                    pill = f"<span class='sm'>🟡 Medium</span>"
                else:
                    pill = f"<span class='sl'>🟢 Low</span>"

                rows += (
                    f"<tr>"
                    f"<td><code style='font-family:JetBrains Mono,monospace;font-size:.75rem;'>"
                    f"{ev['start']:.1f}s – {ev['end']:.1f}s</code></td>"
                    f"<td>{ev['label']}</td>"
                    f"<td>{pill}</td>"
                    f"<td style='opacity:.6;font-size:.72rem;'>{ev['confidence']*100:.0f}%</td>"
                    f"</tr>"
                )

            st.markdown(
                f"<table class='aq-tbl'>"
                f"<thead><tr><th>Timestamp</th><th>Issue</th><th>Severity</th><th>Conf</th></tr></thead>"
                f"<tbody>{rows}</tbody>"
                f"</table>",
                unsafe_allow_html=True,
            )

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    ec1, ec2, ec3 = st.columns([1, 1, 4])

    with ec1:
        payload = json.dumps([
            {"file": r.file_path, "decision": r.overall_decision,
             "severity": r.total_severity,
             "duration_sec": r.metadata.get("duration_sec", 0),
             "events": r.events}
            for r in results
        ], indent=2)
        st.download_button("📋 JSON", data=payload,
                           file_name="audit_report.json",
                           mime="application/json",
                           use_container_width=True)

    with ec2:
        hdr_row = ["file","decision","severity","start","end","issue","conf"]
        rows_csv = [hdr_row]
        for r in results:
            for ev in r.events:
                rows_csv.append([
                    Path(r.file_path).name,
                    r.overall_decision,
                    f"{r.total_severity:.1f}",
                    f"{ev['start']:.1f}",
                    f"{ev['end']:.1f}",
                    ev["label"],
                    f"{ev['confidence']*100:.0f}%",
                ])
        buf = io.StringIO()
        for row in rows_csv:
            buf.write(",".join(f'"{c}"' for c in row) + "\n")
        st.download_button("📊 CSV", data=buf.getvalue(),
                           file_name="audit_report.csv",
                           mime="text/csv",
                           use_container_width=True)
