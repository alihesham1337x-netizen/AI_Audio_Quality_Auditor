import { useEffect, useRef, useState } from "react";

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export default function Waveform({ file, events = [], activeEvent = null }) {
  const canvasRef = useRef(null);
  const audioRef = useRef(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (!file) {
      return;
    }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!file || !canvasRef.current) {
      return;
    }

    const reader = new FileReader();
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioCtx.decodeAudioData(arrayBuffer);
        const channelData = decoded.getChannelData(0);
        setDuration(decoded.duration);
        const width = canvas.width;
        const height = canvas.height;
        const step = Math.ceil(channelData.length / width);
        const amp = height / 2;

        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#e5e7eb";
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = "#2563eb";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let i = 0; i < width; i += 1) {
          const start = i * step;
          let min = 1.0;
          let max = -1.0;
          for (let j = 0; j < step && start + j < channelData.length; j += 1) {
            const value = channelData[start + j];
            min = Math.min(value, min);
            max = Math.max(value, max);
          }
          const y1 = (1 + min) * amp;
          const y2 = (1 + max) * amp;
          ctx.moveTo(i, y1);
          ctx.lineTo(i, y2);
        }
        ctx.stroke();

        if (activeEvent && decoded.duration) {
          const startX = clamp((activeEvent.start / decoded.duration) * width, 0, width);
          const endX = clamp((activeEvent.end / decoded.duration) * width, 0, width);
          ctx.fillStyle = "rgba(248, 113, 113, 0.25)";
          ctx.fillRect(startX, 0, Math.max(1, endX - startX), height);
        }
      } catch (error) {
        // ignore waveform decode errors
      }
    };
    reader.readAsArrayBuffer(file);
  }, [file, activeEvent, events]);

  useEffect(() => {
    if (audioRef.current && activeEvent) {
      audioRef.current.currentTime = activeEvent.start;
    }
  }, [activeEvent]);

  return (
    <div className="waveform-card">
      <div className="waveform-header">
        <strong>Playback preview</strong>
        {duration ? <span>{duration.toFixed(1)}s</span> : null}
      </div>
      <canvas ref={canvasRef} width={860} height={140} className="waveform-canvas" />
      {audioUrl ? <audio controls ref={audioRef} src={audioUrl} className="waveform-player" /> : null}
    </div>
  );
}
