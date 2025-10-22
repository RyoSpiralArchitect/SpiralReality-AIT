# SpiralReality AIT Local Demo Guide

This guide shows how to explore the real-time boundary inference pipeline.

## 1. Launch the demo stack

```bash
docker compose up --build
```

The command builds both the Python diagnostics backend and the Vite dashboard. Once the
containers are ready the services will be available at:

- Backend API and WebSocket: <http://localhost:8000>
- Diagnostics dashboard: <http://localhost:5173>

## 2. Interact with the dashboard

Open the dashboard in your browser and enter any sentence or paragraph. The UI
will display:

- **Boundary segmentation** as a list of detected spans.
- **Gate trace sparkline** showing the latest inference run.
- **Mask energy and attention strength** summary metrics.
- **Boundary probabilities** per index for further inspection.

The dashboard streams its data from the WebSocket endpoint exposed by the
backend service.

## 3. Use the WebSocket programmatically

The backend exposes a `ws://localhost:8000/ws` endpoint. Send a JSON payload of
`{"text": "your sentence"}` and wait for the `result` message. The response
contains the segmentation, boundary probabilities and gate diagnostics.

See [`websocket_demo.ipynb`](./websocket_demo.ipynb) for a Python example.
