import { useEffect, useMemo, useRef, useState } from "react";

type Diagnostics = {
  gate_trace: number[];
  attention_strength: number[];
  mask_energy: number;
};

type Encoding = {
  boundary_probabilities: number[];
  gate_trace: number[];
  phase_local?: number[][];
  gate_mask?: number[][];
  embedding?: number[][];
};

type WsMessage =
  | { type: "ready"; message: string }
  | { type: "processing"; length: number }
  | {
      type: "result";
      text: string;
      segments: string[];
      diagnostics: Diagnostics;
      encoding: Encoding;
    }
  | { type: "error"; message: string };

const WS_URL = (import.meta.env.VITE_WS_URL as string) || `ws://${window.location.hostname}:8000/ws`;

export default function App() {
  const socketRef = useRef<WebSocket | null>(null);
  const [status, setStatus] = useState<string>("Connecting to backend…");
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [text, setText] = useState<string>("Spiral Reality lets models learn reliable boundaries in a single pass.");
  const [segments, setSegments] = useState<string[]>([]);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [encoding, setEncoding] = useState<Encoding | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  useEffect(() => {
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      setStatus("Ready");
      setIsConnected(true);
    };
    socket.onclose = () => {
      setStatus("Disconnected");
      setIsConnected(false);
    };
    socket.onerror = () => setError("WebSocket error. Please refresh.");

    socket.onmessage = (evt) => {
      try {
        const payload = JSON.parse(evt.data) as WsMessage;
        if (payload.type === "ready") {
          setStatus("Ready");
          setError(null);
        } else if (payload.type === "processing") {
          setStatus(`Processing ${payload.length} characters…`);
          setError(null);
        } else if (payload.type === "result") {
          setStatus("Result received");
          setSegments(payload.segments);
          setDiagnostics(payload.diagnostics);
          setEncoding(payload.encoding);
          setError(null);
          setLastUpdated(Date.now());
        } else if (payload.type === "error") {
          setError(payload.message);
        }
      } catch (err) {
        console.error("Invalid message", err);
      }
    };

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, []);

  const boundaryPairs = useMemo(() => {
    if (!encoding) return [];
    return encoding.boundary_probabilities.map((prob, idx) => ({
      index: idx,
      probability: prob,
    }));
  }, [encoding]);

  const gateTraceStats = useMemo(() => {
    if (!diagnostics?.gate_trace?.length) return null;
    const values = diagnostics.gate_trace;
    const total = values.reduce((sum, value) => sum + value, 0);
    const mean = total / values.length;
    const max = Math.max(...values);
    const min = Math.min(...values);
    const variance =
      values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / Math.max(values.length - 1, 1);
    const std = Math.sqrt(Math.max(variance, 0));
    return { mean, max, min, std };
  }, [diagnostics?.gate_trace]);

  const topBoundaries = useMemo(() => {
    if (!boundaryPairs.length) return [];
    return [...boundaryPairs]
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5)
      .map((item) => ({
        ...item,
        probabilityLabel: `${Math.round(item.probability * 1000) / 10}%`,
      }));
  }, [boundaryPairs]);

  const handlePreset = (preset: string) => {
    setText(preset);
  };

  const sendText = () => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setError("Socket not ready yet");
      return;
    }
    socket.send(JSON.stringify({ text }));
  };

  return (
    <div className="page">
      <header>
        <h1>Gate Diagnostics Monitor</h1>
        <div className="status-block">
          <span className={`status-indicator ${isConnected ? "online" : "offline"}`} aria-hidden="true" />
          <p className="status">{status}</p>
          {lastUpdated && (
            <time dateTime={new Date(lastUpdated).toISOString()} className="timestamp">
              Updated {new Date(lastUpdated).toLocaleTimeString()}
            </time>
          )}
        </div>
      </header>

      <section className="input-panel">
        <label htmlFor="text-input">Input text</label>
        <textarea
          id="text-input"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={4}
        />
        <div className="input-actions">
          <button onClick={sendText}>Run inference</button>
          <div className="presets">
            <button type="button" onClick={() => handlePreset("The model anticipates a boundary near each clause transition.")}
              className="preset">
              Clause sample
            </button>
            <button type="button" onClick={() => handlePreset("A long-form paragraph invites more nuanced segmentation with soft boundaries.")}
              className="preset">
              Paragraph sample
            </button>
          </div>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      {segments.length > 0 && (
        <section className="segments">
          <h2>Boundary segmentation</h2>
          <div className="segment-list">
            {segments.map((segment, idx) => (
              <span key={`${segment}-${idx}`} className="segment">
                {segment}
              </span>
            ))}
          </div>
          {topBoundaries.length > 0 && (
            <div className="summary-grid">
              {topBoundaries.map((boundary) => (
                <SummaryCard
                  key={boundary.index}
                  label={`Top boundary #${boundary.index}`}
                  value={boundary.probabilityLabel}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {diagnostics && (
        <section className="diagnostics">
          <h2>Gate trace</h2>
          <Sparkline values={diagnostics.gate_trace} color="var(--accent)" />
          <div className="metrics">
            <Metric label="Mask energy" value={diagnostics.mask_energy.toFixed(3)} />
            <Metric
              label="Attention strength"
              value={diagnostics.attention_strength.map((v) => v.toFixed(3)).join(", ")}
            />
            {gateTraceStats && (
              <>
                <Metric label="Gate avg" value={gateTraceStats.mean.toFixed(3)} />
                <Metric label="Gate σ" value={gateTraceStats.std.toFixed(3)} />
                <Metric label="Extrema" value={`${gateTraceStats.min.toFixed(3)} → ${gateTraceStats.max.toFixed(3)}`} />
              </>
            )}
          </div>
        </section>
      )}

      {boundaryPairs.length > 0 && (
        <section className="probabilities">
          <h2>Boundary probabilities</h2>
          <table>
            <thead>
              <tr>
                <th>Index</th>
                <th>Probability</th>
              </tr>
            </thead>
            <tbody>
              {boundaryPairs.map((item) => (
                <tr key={item.index}>
                  <td>{item.index}</td>
                  <td>
                    <div className="progress">
                      <div className="fill" style={{ width: `${Math.round(item.probability * 100)}%` }} />
                      <span>{item.probability.toFixed(3)}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      {encoding && (
        <section className="encoding">
          <h2>Encoding insights</h2>
          <div className="encoding-grid">
            {encoding.phase_local && encoding.phase_local.length > 0 && (
              <Heatmap
                matrix={encoding.phase_local}
                title="Phase local"
                caption="Phase activations across layers"
              />
            )}
            {encoding.gate_mask && encoding.gate_mask.length > 0 && (
              <Heatmap
                matrix={encoding.gate_mask}
                title="Gate mask"
                caption="Mask attention energy"
              />
            )}
            {encoding.embedding && encoding.embedding.length > 0 && (
              <Heatmap
                matrix={encoding.embedding}
                title="Embedding snapshot"
                caption="Token embeddings projected"
              />
            )}
          </div>
        </section>
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-card">
      <span className="summary-label">{label}</span>
      <span className="summary-value">{value}</span>
    </div>
  );
}

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (!values.length) return <p>No gate trace available.</p>;
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100;
      const y = ((value - min) / Math.max(max - min, 1e-6)) * 100;
      return `${x},${100 - y}`;
    })
    .join(" ");
  const areaPoints = `0,100 ${points} 100,100`;
  const gradientId = useMemo(
    () => `sparklineGradient-${Math.random().toString(36).slice(2, 9)}`,
    []
  );
  return (
    <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none">
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor={color} stopOpacity={0.45} />
          <stop offset="100%" stopColor={color} stopOpacity={0.05} />
        </linearGradient>
      </defs>
      <polygon points={areaPoints} fill={`url(#${gradientId})`} />
      <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
    </svg>
  );
}

function Heatmap({
  matrix,
  title,
  caption,
}: {
  matrix: number[][];
  title: string;
  caption?: string;
}) {
  const normalizedMatrix = useMemo(() => {
    if (!matrix.length) return [];
    const flat = matrix.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    const range = Math.max(max - min, 1e-6);
    return matrix.map((row) => row.map((value) => ({
      value,
      normalized: (value - min) / range,
    })));
  }, [matrix]);

  if (!normalizedMatrix.length) return null;

  return (
    <div className="heatmap">
      <div className="heatmap-header">
        <h3>{title}</h3>
        {caption && <p className="heatmap-caption">{caption}</p>}
      </div>
      <div
        className="heatmap-grid"
        style={{ gridTemplateColumns: `repeat(${normalizedMatrix[0].length}, minmax(0, 1fr))` }}
      >
        {normalizedMatrix.map((row, rowIndex) =>
          row.map((cell, colIndex) => {
            const hue = 220 - cell.normalized * 160;
            const lightness = 25 + cell.normalized * 50;
            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="heatmap-cell"
                style={{ backgroundColor: `hsl(${hue}, 80%, ${lightness}%)` }}
                title={`(${rowIndex}, ${colIndex}) → ${cell.value.toFixed(3)}`}
              />
            );
          })
        )}
      </div>
    </div>
  );
}
