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
  const [text, setText] = useState<string>("Spiral Reality lets models learn reliable boundaries in a single pass.");
  const [segments, setSegments] = useState<string[]>([]);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [encoding, setEncoding] = useState<Encoding | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => setStatus("Ready");
    socket.onclose = () => setStatus("Disconnected");
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
        <p className="status">{status}</p>
      </header>

      <section className="input-panel">
        <label htmlFor="text-input">Input text</label>
        <textarea
          id="text-input"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={4}
        />
        <button onClick={sendText}>Run inference</button>
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
  return (
    <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none">
      <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
    </svg>
  );
}
