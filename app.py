import io
import json
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go
import requests
from streamlit_lottie import st_lottie

from src.audio_auditor.pipeline import AudioAuditPipeline
from src.audio_auditor.audio_loader import safe_load_audio

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
        bg = "linear-gradient(135deg,#0a0f1e 0%,#0d1530 60%,#080d1a 100%)"
        card = "rgba(13,20,40,0.88)"; bdr = "rgba(99,102,241,0.22)"
        t1 = "#e2e8f0"; t2 = "#94a3b8"; acc = "#6366f1"; acc2 = "#8b5cf6"
        glow = "rgba(99,102,241,0.3)"; mbg = "rgba(99,102,241,0.1)"
        pc = "#22c55e"; wc = "#f59e0b"; fc = "#ef4444"
        th = "rgba(99,102,241,0.18)"; tr = "rgba(13,20,40,0.55)"; ta = "rgba(25,35,60,0.55)"
    else:
        bg = "linear-gradient(135deg,#f0f4ff 0%,#eaedff 60%,#f5f0ff 100%)"
        card = "rgba(255,255,255,0.93)"; bdr = "rgba(99,102,241,0.18)"
        t1 = "#1e293b"; t2 = "#475569"; acc = "#4f46e5"; acc2 = "#7c3aed"
        glow = "rgba(99,102,241,0.18)"; mbg = "rgba(99,102,241,0.07)"
        pc = "#16a34a"; wc = "#d97706"; fc = "#dc2626"
        th = "rgba(99,102,241,0.1)"; tr = "rgba(255,255,255,0.8)"; ta = "rgba(240,244,255,0.8)"

    st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box}}
.stApp,.main,.block-container{{background:{bg}!important;font-family:'Inter',sans-serif!important;color:{t1}!important}}
.block-container{{padding:1rem 1.8rem 2rem!important;max-width:1380px!important}}
.stApp::before{{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 55% 38% at 18% 8%,{glow},transparent),
             radial-gradient(ellipse 45% 32% at 82% 82%,rgba(139,92,246,.15),transparent);
  animation:orb 22s ease-in-out infinite alternate}}
@keyframes orb{{0%{{transform:translate(0,0) scale(1)}}100%{{transform:translate(25px,-18px) scale(1.03)}}}}

/* header */
.aq-hdr{{display:flex;align-items:center;gap:.75rem;padding:.5rem 0 .25rem}}
.aq-logo{{width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,{acc},{acc2});
  display:flex;align-items:center;justify-content:center;font-size:1.3rem;
  box-shadow:0 6px 20px {glow};animation:pulse 3s ease-in-out infinite}}
@keyframes pulse{{0%,100%{{box-shadow:0 6px 20px {glow}}}50%{{box-shadow:0 6px 32px {acc}99}}}}
.aq-title{{font-size:1.7rem;font-weight:800;color:{t1};line-height:1.1}}
.aq-sub{{font-size:.82rem;color:{t2};margin-top:1px}}
.aq-chips{{display:flex;flex-wrap:wrap;gap:6px;margin:.5rem 0 .75rem}}
.aq-chip{{padding:3px 10px;border-radius:999px;font-size:.75rem;font-weight:500;
  background:{mbg};border:1px solid {bdr};color:{t2}}}

/* buttons */
.stButton>button,.stDownloadButton>button{{
  background:linear-gradient(135deg,{acc},{acc2})!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-weight:600!important;
  padding:.45rem 1.2rem!important;font-size:.85rem!important;
  box-shadow:0 3px 12px {glow}!important;transition:all .2s!important}}
.stButton>button:hover,.stDownloadButton>button:hover{{transform:translateY(-1px)!important;box-shadow:0 6px 20px {glow}!important}}

/* metrics */
[data-testid="stMetric"]{{background:{mbg}!important;border:1px solid {bdr}!important;border-radius:10px!important;padding:.6rem .9rem!important}}
[data-testid="stMetricLabel"]{{color:{t2}!important;font-size:.72rem!important;font-weight:500!important}}
[data-testid="stMetricValue"]{{color:{t1}!important;font-size:1.3rem!important;font-weight:700!important}}

/* file uploader */
.stFileUploader{{background:transparent!important;border:2px dashed {bdr}!important;border-radius:12px!important;padding:.6rem!important}}
.stFileUploader:hover{{border-color:{acc}!important}}

/* expander */
.streamlit-expanderHeader{{background:{card}!important;border-radius:10px!important;border:1px solid {bdr}!important;font-weight:600!important;color:{t1}!important;padding:.5rem .8rem!important}}
.streamlit-expanderContent{{background:{card}!important;border:1px solid {bdr}!important;border-top:none!important;border-radius:0 0 10px 10px!important;padding:.75rem!important}}

/* badges */
.b-pass{{background:{pc}1a;color:{pc};border:1px solid {pc}44;border-radius:6px;padding:2px 10px;font-weight:700;font-size:.82rem}}
.b-warn{{background:{wc}1a;color:{wc};border:1px solid {wc}44;border-radius:6px;padding:2px 10px;font-weight:700;font-size:.82rem}}
.b-fail{{background:{fc}1a;color:{fc};border:1px solid {fc}44;border-radius:6px;padding:2px 10px;font-weight:700;font-size:.82rem}}

/* event table */
.aq-tbl{{width:100%;border-collapse:collapse;font-size:.82rem;border-radius:10px;overflow:hidden}}
.aq-tbl thead tr{{background:{th}}}
.aq-tbl th{{padding:7px 12px;text-align:left;font-weight:600;color:{t2};font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}}
.aq-tbl tbody tr:nth-child(odd){{background:{tr}}}
.aq-tbl tbody tr:nth-child(even){{background:{ta}}}
.aq-tbl td{{padding:6px 12px;color:{t1};border-bottom:1px solid {bdr}}}
.aq-tbl tbody tr:hover{{background:{mbg}!important}}

/* severity pills */
.sc{{background:#ef444418;color:#ef4444;border:1px solid #ef444440;border-radius:5px;padding:1px 7px;font-size:.72rem;font-weight:600}}
.sh{{background:#f9731618;color:#f97316;border:1px solid #f9731640;border-radius:5px;padding:1px 7px;font-size:.72rem;font-weight:600}}
.sm{{background:#eab30818;color:#eab308;border:1px solid #eab30840;border-radius:5px;padding:1px 7px;font-size:.72rem;font-weight:600}}
.sl{{background:#22c55e18;color:#22c55e;border:1px solid #22c55e40;border-radius:5px;padding:1px 7px;font-size:.72rem;font-weight:600}}

/* section label */
.aq-lbl{{font-size:.78rem;font-weight:600;color:{t2};text-transform:uppercase;letter-spacing:.06em;margin:.8rem 0 .3rem;display:flex;align-items:center;gap:5px}}

/* divider */
hr{{border-color:{bdr}!important;margin:.8rem 0!important}}
audio{{border-radius:8px!important;width:100%!important;height:36px!important}}
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-thumb{{background:{bdr};border-radius:3px}}
.stTabs [data-baseweb="tab-list"]{{background:{mbg}!important;border-radius:10px!important;padding:3px!important;gap:3px!important}}
.stTabs [data-baseweb="tab"]{{border-radius:7px!important;font-weight:500!important;color:{t2}!important;font-size:.82rem!important;padding:.3rem .7rem!important}}
.stTabs [aria-selected="true"]{{background:{acc}!important;color:#fff!important}}
.stProgress>div>div{{background:linear-gradient(90deg,{acc},{acc2})!important;border-radius:3px!important}}
#MainMenu,footer,header{{visibility:hidden}}
</style>""", unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ── Header ────────────────────────────────────────────────────────────────────
hdr_col, toggle_col = st.columns([6, 1])
with hdr_col:
    st.markdown("""
    <div class='aq-hdr'>
        <div class='aq-logo'>🎙️</div>
        <div>
            <div class='aq-title'>AI Audio Quality Auditor</div>
            <div class='aq-sub'>Background noise detection for call-centre recordings</div>
        </div>
    </div>
    <div class='aq-chips'>
        <span class='aq-chip'>🎵 MP3 &amp; WAV</span>
        <span class='aq-chip'>🧠 Silero VAD</span>
        <span class='aq-chip'>📊 Noise-floor analysis</span>
        <span class='aq-chip'>⚡ Transient detection</span>
        <span class='aq-chip'>📥 JSON &amp; CSV export</span>
    </div>
    """, unsafe_allow_html=True)
with toggle_col:
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    if st.button("☀️" if st.session_state.dark_mode else "🌙", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.divider()

# ── Upload + Config (compact 2-col) ──────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="medium")

with left_col:
    st.markdown("<div class='aq-lbl'>📁 Upload Files</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "files", type=["mp3", "wav"], accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        for f in uploaded_files:
            kb = len(f.getvalue()) / 1024
            st.markdown(
                f"<div style='font-size:.78rem;opacity:.75;padding:1px 0;'>▸ {f.name[:38]} "
                f"<span style='opacity:.55'>({kb:.0f} KB)</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='aq-lbl' style='margin-top:.7rem;'>🎚️ Sensitivity</div>", unsafe_allow_html=True)
    sensitivity = st.slider("sens", 0, 100, st.session_state.sensitivity,
                            label_visibility="collapsed")
    st.session_state.sensitivity = sensitivity
    sens_txt = "Conservative" if sensitivity < 35 else ("Aggressive" if sensitivity > 65 else "Balanced")
    st.markdown(
        f"<div style='font-size:.72rem;opacity:.6;margin-top:-.2rem;'>{sensitivity}% — {sens_txt}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    audit_clicked = st.button("🚀 Run Audit", type="primary",
                              use_container_width=True, disabled=not uploaded_files)
    if LOTTIE:
        st_lottie(LOTTIE, height=90, key="lottie_side", speed=0.8)

with right_col:
    st.markdown("<div class='aq-lbl'>🎧 Previews</div>", unsafe_allow_html=True)
    if uploaded_files:
        tabs = st.tabs([f"🎵 {f.name[:24]}{'…' if len(f.name)>24 else ''}"
                        for f in uploaded_files])
        for tab, uf in zip(tabs, uploaded_files):
            with tab:
                fmt = "audio/wav" if uf.name.lower().endswith(".wav") else "audio/mpeg"
                st.audio(uf, format=fmt)
    else:
        st.markdown(
            "<div style='text-align:center;padding:2rem .5rem;opacity:.5;font-size:.85rem;'>"
            "🎙️ Upload files to preview them here</div>",
            unsafe_allow_html=True,
        )

# ── Audit processing ──────────────────────────────────────────────────────────
if audit_clicked and uploaded_files:
    st.session_state.results = None
    pipeline   = AudioAuditPipeline()
    upload_dir = Path("./temp_uploads")
    upload_dir.mkdir(exist_ok=True)

    st.divider()
    prog  = st.progress(0)
    stxt  = st.empty()

    t = sensitivity / 100.0
    z_thr  = max(1.8, 3.2 - t * 1.4)
    e_thr  = max(0.010, 0.030 - t * 0.020)
    f_min  = max(0.10,  0.22  - t * 0.12)

    results = []
    for idx, uf in enumerate(uploaded_files):
        stxt.markdown(
            f"<div style='font-size:.82rem;opacity:.7;'>⚙️ Analysing "
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
        "<div style='font-size:.82rem;color:#22c55e;'>✅ Audit complete</div>",
        unsafe_allow_html=True,
    )

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    st.divider()

    # ── Summary bar ──────────────────────────────────────────────────────────
    total_ev  = sum(len(r.events) for r in results)
    avg_sev   = sum(r.total_severity for r in results) / len(results)
    n_fail    = sum(1 for r in results if r.overall_decision == "FAIL")
    n_warn    = sum(1 for r in results if r.overall_decision == "WARNING")
    n_pass    = sum(1 for r in results if r.overall_decision == "PASS")

    mc = st.columns(5)
    mc[0].metric("Files",        len(results))
    mc[1].metric("Events",       total_ev)
    mc[2].metric("Avg Severity", f"{avg_sev:.1f}")
    mc[3].metric("⛔ FAIL",      n_fail)
    mc[4].metric("⚠️ WARN",      n_warn)

    st.divider()

    # ── Per-file cards ────────────────────────────────────────────────────────
    BADGE = {"PASS": "b-pass", "WARNING": "b-warn", "FAIL": "b-fail"}
    ICON  = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "⛔"}
    CMAP  = {
        "critical": ("rgba(239,68,68,.22)",  "rgba(239,68,68,.85)"),
        "high":     ("rgba(249,115,22,.2)",  "rgba(249,115,22,.85)"),
        "medium":   ("rgba(234,179,8,.18)",  "rgba(234,179,8,.85)"),
        "low":      ("rgba(34,197,94,.15)",  "rgba(34,197,94,.85)"),
    }

    for result in results:
        fname = Path(result.file_path).name
        dec   = result.overall_decision
        icon  = ICON.get(dec, "❓")
        bcls  = BADGE.get(dec, "b-warn")

        with st.expander(f"{icon}  {fname}", expanded=(dec != "PASS")):

            # ── compact header row ────────────────────────────────────────
            hc = st.columns([2, 1, 1, 1])
            hc[0].markdown(f"<span class='{bcls}'>{icon} {dec}</span>",
                           unsafe_allow_html=True)
            hc[1].metric("Severity", f"{result.total_severity:.1f}")
            hc[2].metric("Events",   len(result.events))
            hc[3].metric("Duration", f"{result.metadata.get('duration_sec',0):.1f}s")

            if not result.events:
                st.markdown(
                    "<div style='text-align:center;padding:1.2rem;font-size:.85rem;"
                    "color:#22c55e;'>✅ No violations detected</div>",
                    unsafe_allow_html=True,
                )
                continue

            # ── events table ─────────────────────────────────────────────
            st.markdown("<div class='aq-lbl' style='margin-top:.6rem;'>🔍 Events</div>",
                        unsafe_allow_html=True)
            rows = ""
            for ev in result.events:
                s = ev["severity"]
                if s >= 8:
                    pill = f"<span class='sc'>🔴 {s:.1f}</span>"
                elif s >= 6:
                    pill = f"<span class='sh'>🟠 {s:.1f}</span>"
                elif s >= 4.5:
                    pill = f"<span class='sm'>🟡 {s:.1f}</span>"
                else:
                    pill = f"<span class='sl'>🟢 {s:.1f}</span>"
                rows += (
                    f"<tr>"
                    f"<td><code style='font-family:JetBrains Mono,monospace;font-size:.78rem'>"
                    f"{ev['start']:.2f}s</code></td>"
                    f"<td><code style='font-family:JetBrains Mono,monospace;font-size:.78rem'>"
                    f"{ev['end']:.2f}s</code></td>"
                    f"<td><strong>{ev['label']}</strong></td>"
                    f"<td>{pill}</td>"
                    f"<td style='opacity:.7'>{ev['confidence']*100:.0f}%</td>"
                    f"</tr>"
                )
            st.markdown(
                f"<table class='aq-tbl'><thead><tr>"
                f"<th>Start</th><th>End</th><th>Label</th>"
                f"<th>Severity</th><th>Conf</th>"
                f"</tr></thead><tbody>{rows}</tbody></table>",
                unsafe_allow_html=True,
            )

            # ── waveform (compact height=140) ─────────────────────────────
            st.markdown("<div class='aq-lbl' style='margin-top:.7rem;'>📈 Waveform</div>",
                        unsafe_allow_html=True)
            try:
                wv, sr = safe_load_audio(result.file_path)
                ag = wv[:, 0] if wv.ndim == 2 else wv
                step   = max(1, len(ag) // 3500)
                t_ax   = [i / sr for i in range(0, len(ag), step)]
                y_vals = ag[::step].tolist()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t_ax, y=y_vals, mode="lines",
                    line=dict(color="#6366f1", width=0.8),
                    hovertemplate="t=%{x:.2f}s<extra></extra>",
                ))
                for ev in result.events:
                    s = ev["severity"]
                    tier = ("critical" if s >= 8 else
                            "high"     if s >= 6 else
                            "medium"   if s >= 4.5 else "low")
                    fc, lc = CMAP[tier]
                    fig.add_vrect(
                        x0=ev["start"], x1=ev["end"],
                        fillcolor=fc, opacity=1, layer="below",
                        line_width=1, line_color=lc,
                    )
                is_dark = st.session_state.dark_mode
                ax_col  = "#64748b" if is_dark else "#94a3b8"
                fig.update_layout(
                    height=140,
                    margin=dict(l=0, r=0, t=4, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="", color=ax_col,
                               gridcolor="rgba(148,163,184,.08)", showgrid=True,
                               tickfont=dict(size=9)),
                    yaxis=dict(title="", color=ax_col,
                               gridcolor="rgba(148,163,184,.08)", showgrid=True,
                               tickfont=dict(size=9)),
                    showlegend=False, hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            except Exception as e:
                st.caption(f"Waveform unavailable: {e}")

            # ── single playback bar ───────────────────────────────────────
            st.markdown("<div class='aq-lbl'>🔊 Playback</div>",
                        unsafe_allow_html=True)
            matched = next((uf for uf in (uploaded_files or [])
                            if uf.name == fname), None)
            if matched:
                fmt = "audio/wav" if fname.lower().endswith(".wav") else "audio/mpeg"
                st.audio(matched, format=fmt)
            else:
                st.caption("Original file not in current upload session.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("<div class='aq-lbl'>📥 Export</div>", unsafe_allow_html=True)
    ec1, ec2 = st.columns(2)

    with ec1:
        payload = json.dumps([
            {"file": r.file_path, "decision": r.overall_decision,
             "severity": r.total_severity,
             "duration_sec": r.metadata.get("duration_sec", 0),
             "events": r.events}
            for r in results
        ], indent=2)
        st.download_button("📋 Export JSON", data=payload,
                           file_name="audit_report.json",
                           mime="application/json",
                           use_container_width=True)

    with ec2:
        hdr = ["file_path","decision","total_severity","start","end",
               "label","severity","confidence"]
        rows_csv = [hdr]
        for r in results:
            for ev in r.events:
                rows_csv.append([
                    r.file_path, r.overall_decision,
                    f"{r.total_severity:.2f}",
                    f"{ev['start']:.2f}", f"{ev['end']:.2f}",
                    ev["label"], f"{ev['severity']:.1f}",
                    f"{ev['confidence']:.2f}",
                ])
        buf = io.StringIO()
        for row in rows_csv:
            buf.write(",".join(f'"{c}"' for c in row) + "\n")
        st.download_button("📊 Export CSV", data=buf.getvalue(),
                           file_name="audit_report.csv",
                           mime="text/csv",
                           use_container_width=True)
