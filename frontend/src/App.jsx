import { useEffect, useState } from "react";
import axios from "axios";
import Waveform from "./Waveform";

function AudioPreview({ file }) {
  const [audioUrl, setAudioUrl] = useState(null);

  useEffect(() => {
    if (!file) {
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => {
      URL.revokeObjectURL(url);
      setAudioUrl(null);
    };
  }, [file]);

  if (!audioUrl) {
    return null;
  }

  return <audio controls src={audioUrl} className="audio-player" />;
}

function App() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [selectedEvents, setSelectedEvents] = useState({});
  const [noiseSensitivity, setNoiseSensitivity] = useState(50);
  const [isDarkMode, setIsDarkMode] = useState(false);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("audioAuditorDarkMode");
    if (storedTheme) {
      setIsDarkMode(storedTheme === "true");
    }
  }, []);

  const toggleDarkMode = () => {
    setIsDarkMode((prev) => {
      const next = !prev;
      window.localStorage.setItem("audioAuditorDarkMode", next.toString());
      return next;
    });
  };

  const handleFiles = (event) => {
    setFiles(Array.from(event.target.files));
  };

  const downloadReport = (filename, content, type) => {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const exportResultsJSON = () => {
    if (!results) return;
    const payload = JSON.stringify(results, null, 2);
    downloadReport("audit_report.json", payload, "application/json");
  };

  const exportResultsCSV = () => {
    if (!results) return;
    const rows = [
      [
        "file_path",
        "overall_decision",
        "total_severity",
        "start",
        "end",
        "label",
        "severity",
        "confidence",
      ],
    ];
    results.forEach((result) => {
      result.events.forEach((event) => {
        rows.push([
          result.file_path,
          result.overall_decision,
          result.total_severity.toFixed(2),
          event.start.toFixed(2),
          event.end.toFixed(2),
          event.label,
          event.severity.toFixed(1),
          event.confidence.toFixed(2),
        ]);
      });
    });
    const csv = rows.map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(",")).join("\n");
    downloadReport("audit_report.csv", csv, "text/csv;charset=utf-8;");
  };

  const submitFiles = async () => {
    if (!files.length) {
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("sensitivity", (noiseSensitivity / 100).toString());

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post("/api/audit", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const selectEvent = (filePath, index) => {
    setSelectedEvents((prev) => ({ ...prev, [filePath]: index }));
  };

  const getUploadedFile = (result) => {
    const fileName = result.file_path.split("/").pop().split("\\").pop();
    return files.find((file) => file.name === fileName);
  };

  const severityClass = (severity) => {
    if (severity >= 8) return "severity-critical";
    if (severity >= 6) return "severity-high";
    if (severity >= 4) return "severity-medium";
    return "severity-low";
  };

  const totalEvents = results?.reduce((sum, result) => sum + result.events.length, 0) ?? 0;

  return (
    <div className={`app-container${isDarkMode ? " dark-mode" : ""}`}>
      <header>
        <div className="header-row">
          <div>
            <h1>AI Audio Quality Auditor</h1>
            <p>Upload MP3/WAV call recordings for automated noise compliance auditing with agent-side noise detection.</p>
          </div>
          <button className="theme-toggle" onClick={toggleDarkMode}>
            {isDarkMode ? "Light mode" : "Dark mode"}
          </button>
        </div>
        <div className="hero-chips">
          <span className="hero-chip">Bulk MP3/WAV uploads</span>
          <span className="hero-chip">Agent-left noise focus</span>
          <span className="hero-chip">Inline playback + export</span>
        </div>
      </header>

      <section className="upload-card">
        <label className="file-input-label">
          Choose audio files
          <input type="file" accept="audio/mp3,audio/wav" multiple onChange={handleFiles} />
        </label>
        {files.length ? (
          <div className="selected-file-list">
            <h3>Selected files</h3>
            {files.map((file, idx) => (
              <div key={idx} className="file-preview">
                <span>{file.name}</span>
                <AudioPreview file={file} />
              </div>
            ))}
          </div>
        ) : null}
        <div className="slider-control">
          <label htmlFor="noise-sensitivity">Background noise sensitivity</label>
          <input
            id="noise-sensitivity"
            type="range"
            min="0"
            max="100"
            value={noiseSensitivity}
            onChange={(event) => setNoiseSensitivity(Number(event.target.value))}
          />
          <span>{noiseSensitivity}%</span>
        </div>
        <button className="primary-button" disabled={!files.length || loading} onClick={submitFiles}>
          {loading ? "Processing..." : "Audit files"}
          {loading ? <span className="loading-spinner" /> : null}
        </button>
        {files.length ? <p className="selected-count">{files.length} file(s) selected.</p> : null}
      </section>

      {error ? <div className="error-card">{error}</div> : null}

      {results ? (
        <section className="results-card">
          <div className="results-header">
            <div>
              <h2>Audit Results</h2>
              <p className="result-summary">{files.length} uploaded file(s) • {totalEvents} event(s) found</p>
            </div>
            <div className="export-buttons">
              <button className="secondary-button" disabled={loading} onClick={exportResultsJSON}>
                Export JSON
              </button>
              <button className="secondary-button" disabled={loading} onClick={exportResultsCSV}>
                Export CSV
              </button>
            </div>
          </div>
          {results.map((result, idx) => {
            const localFile = getUploadedFile(result);
            const activeIndex = selectedEvents[result.file_path] ?? null;
            return (
              <div key={idx} className="result-item">
                <h3>{result.file_path}</h3>
                <p className="decision">Decision: {result.overall_decision}</p>
                <p>Total severity: {result.total_severity.toFixed(1)}</p>
                <div className="event-list">
                  {result.events.length ? (
                    <table>
                      <thead>
                        <tr>
                          <th>Start</th>
                          <th>End</th>
                          <th>Label</th>
                          <th>Severity</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.events.map((event, eidx) => (
                          <tr
                            key={eidx}
                            className={`${severityClass(event.severity)} ${activeIndex === eidx ? "active-event" : ""}`}
                            onClick={() => selectEvent(result.file_path, eidx)}
                          >
                            <td>{event.start.toFixed(2)}s</td>
                            <td>{event.end.toFixed(2)}s</td>
                            <td>{event.label}</td>
                            <td>{event.severity.toFixed(1)}</td>
                            <td>{event.confidence.toFixed(2)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p>No violations detected.</p>
                  )}
                </div>
                {localFile ? (
                  <Waveform file={localFile} events={result.events} activeEvent={result.events[activeIndex]} />
                ) : (
                  <p className="warning-text">Waveform preview unavailable for this file.</p>
                )}
                {localFile ? (
                  <div className="audio-playback">
                    <strong>Play original file</strong>
                    <AudioPreview file={localFile} />
                  </div>
                ) : null}
              </div>
            );
          })}
        </section>
      ) : null}
    </div>
  );
}

export default App;
