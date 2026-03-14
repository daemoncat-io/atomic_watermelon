"""
live_monitor.py

Live training dashboard. Watches the training log JSON and serves
a self-updating HTML page on localhost.

Usage:
    python live_monitor.py                                          # auto-find latest log
    python live_monitor.py checkpoints/*.json       # specific log
    python live_monitor.py checkpoints/*.json 8080  # custom port

Opens in browser. Polls every 30s. No dependencies beyond stdlib.
"""

from urllib.parse import urlparse
import http.server
import webbrowser
import threading
import json
import glob
import sys
import os

LOG_DIR = "checkpoints"
DEFAULT_PORT = 8077


def find_latest_log(log_dir: str = LOG_DIR) -> str | None:
    """Find the most recently modified JSON log in the checkpoints dir."""
    pattern = os.path.join(log_dir, "*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def read_log(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return None


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bridge Transformer — Live Monitor</title>
<style>
  :root {
    --bg: #0a0a0a;
    --panel: #111111;
    --border: #222222;
    --text: #cccccc;
    --dim: #666666;
    --accent: #4fc3f7;
    --green: #66bb6a;
    --red: #ef5350;
    --yellow: #ffd54f;
    --blue: #42a5f5;
    --mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    line-height: 1.5;
    padding: 16px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }

  .header h1 {
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.5px;
  }

  .header .meta {
    font-size: 11px;
    color: var(--dim);
  }

  .status-bar {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    padding: 10px 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 16px;
    font-size: 12px;
  }

  .status-bar .stat {
    display: flex;
    gap: 6px;
  }

  .status-bar .label { color: var(--dim); }
  .status-bar .value { color: var(--text); font-weight: 600; }
  .status-bar .best { color: var(--green); }
  .status-bar .warn { color: var(--yellow); }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-bottom: 16px;
  }

  .grid.full { grid-template-columns: 1fr; }

  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px;
  }

  .panel h2 {
    font-size: 11px;
    font-weight: 600;
    color: var(--dim);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
  }

  canvas {
    width: 100% !important;
    height: 200px !important;
    display: block;
  }

  .sample-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 12px;
  }

  .sample-panel h3 {
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 8px;
  }

  .sample-text {
    font-size: 12px;
    line-height: 1.6;
    color: var(--text);
    white-space: pre-wrap;
    word-wrap: break-word;
    max-height: 200px;
    overflow-y: auto;
    padding: 8px;
    background: #0a0a0a;
    border-radius: 3px;
    border: 1px solid var(--border);
  }

  .sample-text .prompt {
    color: var(--yellow);
    font-weight: 600;
  }

  .epoch-nav {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding: 8px 14px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
  }

  .epoch-nav button {
    background: var(--border);
    color: var(--text);
    border: none;
    padding: 4px 12px;
    border-radius: 3px;
    cursor: pointer;
    font-family: var(--mono);
    font-size: 12px;
  }

  .epoch-nav button:hover { background: #333; }
  .epoch-nav button:disabled { opacity: 0.3; cursor: default; }
  .epoch-nav .current { color: var(--accent); font-weight: 600; }

  .ticker {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 6px 16px;
    font-size: 11px;
    color: var(--dim);
    display: flex;
    justify-content: space-between;
  }

  .ticker .live {
    color: var(--green);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  .no-data {
    text-align: center;
    padding: 80px 20px;
    color: var(--dim);
    font-size: 14px;
  }
</style>
</head>
<body>

<div class="header">
  <h1>BRIDGE TRANSFORMER — LIVE MONITOR</h1>
  <div class="meta" id="logfile"></div>
</div>

<div class="status-bar" id="status-bar">
  <div class="stat"><span class="label">epoch</span><span class="value" id="s-epoch">—</span></div>
  <div class="stat"><span class="label">train</span><span class="value" id="s-train">—</span></div>
  <div class="stat"><span class="label">val</span><span class="value" id="s-val">—</span></div>
  <div class="stat"><span class="label">best</span><span class="value best" id="s-best">—</span></div>
  <div class="stat"><span class="label">grad</span><span class="value" id="s-grad">—</span></div>
  <div class="stat"><span class="label">lr</span><span class="value" id="s-lr">—</span></div>
  <div class="stat"><span class="label">epoch_t</span><span class="value" id="s-time">—</span></div>
  <div class="stat"><span class="label">elapsed</span><span class="value" id="s-elapsed">—</span></div>
  <div class="stat"><span class="label">ram</span><span class="value" id="s-ram">—</span></div>
  <div class="stat"><span class="label">gpu</span><span class="value" id="s-gpu">—</span></div>
</div>

<div class="grid">
  <div class="panel">
    <h2>Loss</h2>
    <canvas id="chart-loss"></canvas>
  </div>
  <div class="panel">
    <h2>Gradient Norm</h2>
    <canvas id="chart-grad"></canvas>
  </div>
</div>

<div class="epoch-nav">
  <button id="btn-prev" onclick="navEpoch(-1)">◀ prev</button>
  <span class="current" id="nav-label">latest</span>
  <button id="btn-next" onclick="navEpoch(1)">next ▶</button>
  <button onclick="navEpoch(Infinity)">latest ▶▶</button>
</div>

<div id="samples-container"></div>

<div class="ticker">
  <span><span class="live">●</span> polling every 30s</span>
  <span id="last-update">—</span>
</div>

<script>
// ================================================================
// STATE
// ================================================================

let DATA = null;
let viewEpochIdx = -1; // -1 = latest

// ================================================================
// POLLING
// ================================================================

async function poll() {
  try {
    const resp = await fetch('/api/data?_=' + Date.now());
    if (resp.ok) {
      DATA = await resp.json();
      render();
      document.getElementById('last-update').textContent =
        'updated: ' + new Date().toLocaleTimeString();
    }
  } catch(e) {}
}

setInterval(poll, 30000);
poll();

// ================================================================
// NAVIGATION
// ================================================================

function navEpoch(delta) {
  if (!DATA || !DATA.epochs || DATA.epochs.length === 0) return;
  const maxIdx = DATA.epochs.length - 1;

  if (delta === Infinity) {
    viewEpochIdx = -1;
  } else if (viewEpochIdx === -1) {
    viewEpochIdx = maxIdx + delta;
  } else {
    viewEpochIdx += delta;
  }

  viewEpochIdx = Math.max(0, Math.min(maxIdx, viewEpochIdx));
  if (viewEpochIdx === maxIdx) viewEpochIdx = -1;

  render();
}

// ================================================================
// RENDER
// ================================================================

function render() {
  if (!DATA) return;

  const epochs = DATA.epochs || [];
  if (epochs.length === 0) {
    document.getElementById('samples-container').innerHTML =
      '<div class="no-data">Waiting for first epoch...</div>';
    return;
  }

  // Which epoch to show samples for
  const idx = viewEpochIdx === -1 ? epochs.length - 1 : viewEpochIdx;
  const current = epochs[idx];
  const latest = epochs[epochs.length - 1];

  // Logfile
  const cfg = DATA.config || {};
  document.getElementById('logfile').textContent =
    (cfg.model || '') + ' | d=' + (cfg.d_model || '?') +
    ' n=' + (cfg.n_layers || '?') + ' h=' + (cfg.n_heads || '?') +
    ' mem=' + (cfg.memory_slots || '?') + ' chunk=' + (cfg.compress_chunk || '?') +
    ' vocab=' + (cfg.vocab_size || '?');

  // Status bar (always shows latest)
  document.getElementById('s-epoch').textContent = latest.epoch;
  document.getElementById('s-train').textContent = latest.train_loss.toFixed(4);
  document.getElementById('s-val').textContent = latest.val_loss.toFixed(4);
  document.getElementById('s-grad').textContent = latest.grad_norm.toFixed(4);
  document.getElementById('s-lr').textContent = latest.lr.toExponential(1);
  document.getElementById('s-time').textContent = latest.epoch_time_sec.toFixed(0) + 's';

  const elapsed = latest.elapsed_sec || 0;
  const hrs = Math.floor(elapsed / 3600);
  const mins = Math.floor((elapsed % 3600) / 60);
  document.getElementById('s-elapsed').textContent = hrs + 'h ' + mins + 'm';

  const best = DATA.best;
  document.getElementById('s-best').textContent =
    best ? best.val_loss.toFixed(4) + ' @' + best.epoch : '—';

  const mem = latest.utilization?.memory;
  document.getElementById('s-ram').textContent =
    mem ? mem.process_ram_gb.toFixed(1) + 'GB' : '—';

  const gpu = latest.utilization?.gpu;
  document.getElementById('s-gpu').textContent =
    gpu ? (gpu.allocated_gb || gpu.driver_allocated_gb || 0).toFixed(1) + 'GB' : '—';

  // Nav label
  const navLabel = viewEpochIdx === -1
    ? 'epoch ' + current.epoch + ' (latest)'
    : 'epoch ' + current.epoch;
  document.getElementById('nav-label').textContent = navLabel;
  document.getElementById('btn-prev').disabled = idx <= 0;
  document.getElementById('btn-next').disabled = viewEpochIdx === -1;

  // Charts
  drawLossChart(epochs);
  drawGradChart(epochs);

  // Samples
  renderSamples(current);
}

// ================================================================
// CHARTS (canvas — no libraries)
// ================================================================

function drawChart(canvasId, datasets, opts = {}) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;

  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const W = rect.width;
  const H = rect.height;
  const pad = { top: 10, right: 60, bottom: 24, left: 8 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  // Find ranges
  let allVals = [];
  for (const ds of datasets) allVals.push(...ds.data);
  allVals = allVals.filter(v => v !== null && v !== undefined && isFinite(v));
  if (allVals.length === 0) return;

  let yMin = opts.yMin !== undefined ? opts.yMin : Math.min(...allVals);
  let yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...allVals);
  if (yMin === yMax) { yMin -= 0.1; yMax += 0.1; }
  const yPad = (yMax - yMin) * 0.05;
  yMin -= yPad;
  yMax += yPad;

  const n = datasets[0].data.length;
  const xScale = plotW / Math.max(n - 1, 1);
  const yScale = plotH / (yMax - yMin);

  function toX(i) { return pad.left + i * xScale; }
  function toY(v) { return pad.top + plotH - (v - yMin) * yScale; }

  // Grid lines
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth = 1;
  const nGrid = 4;
  for (let i = 0; i <= nGrid; i++) {
    const y = pad.top + (plotH / nGrid) * i;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(pad.left + plotW, y);
    ctx.stroke();

    const val = yMax - (yMax - yMin) * (i / nGrid);
    ctx.fillStyle = '#444';
    ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText(val.toFixed(opts.decimals || 4), pad.left + plotW + 4, y + 3);
  }

  // Data lines
  for (const ds of datasets) {
    ctx.strokeStyle = ds.color;
    ctx.lineWidth = ds.width || 1.5;
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < ds.data.length; i++) {
      const v = ds.data[i];
      if (v === null || v === undefined || !isFinite(v)) continue;
      if (!started) { ctx.moveTo(toX(i), toY(v)); started = true; }
      else ctx.lineTo(toX(i), toY(v));
    }
    ctx.stroke();

    // Label at end
    const lastVal = ds.data[ds.data.length - 1];
    if (lastVal !== null && lastVal !== undefined) {
      ctx.fillStyle = ds.color;
      ctx.font = '10px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(ds.label + ' ' + lastVal.toFixed(opts.decimals || 4),
        pad.left + plotW + 4, toY(lastVal) - 8);
    }
  }

  // Epoch axis
  ctx.fillStyle = '#444';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(n / 6));
  for (let i = 0; i < n; i += step) {
    ctx.fillText(datasets[0].epochs ? datasets[0].epochs[i] : i,
      toX(i), H - 4);
  }
}

function drawLossChart(epochs) {
  drawChart('chart-loss', [
    {
      label: 'train',
      data: epochs.map(e => e.train_loss),
      epochs: epochs.map(e => e.epoch),
      color: '#4fc3f7',
      width: 1.5,
    },
    {
      label: 'val',
      data: epochs.map(e => e.val_loss),
      color: '#ffd54f',
      width: 1.5,
    },
  ], { yMin: 0, decimals: 4 });
}

function drawGradChart(epochs) {
  drawChart('chart-grad', [
    {
      label: 'grad',
      data: epochs.map(e => e.grad_norm),
      epochs: epochs.map(e => e.epoch),
      color: '#66bb6a',
      width: 1.5,
    },
  ], { yMin: 0, decimals: 4 });
}

// ================================================================
// SAMPLES
// ================================================================

function renderSamples(epoch) {
  const container = document.getElementById('samples-container');
  const samples = epoch.samples;

  if (!samples) {
    container.innerHTML = '<div class="no-data">No samples for this epoch</div>';
    return;
  }

  // Group: fixed prompts first, then random
  const fixed = [];
  const random = [];

  for (const [name, text] of Object.entries(samples)) {
    if (name.startsWith('random_')) {
      random.push({ name, text });
    } else {
      fixed.push({ name, text });
    }
  }

  const all = [...fixed, ...random];
  let html = '<div class="grid">';

  for (const { name, text } of all) {
    // Try to split prompt from generation at a natural boundary
    const promptEnd = findPromptBoundary(name, text);
    const promptPart = text.substring(0, promptEnd);
    const genPart = text.substring(promptEnd);

    html += '<div class="sample-panel">';
    html += '<h3>' + escHtml(name) + '</h3>';
    html += '<div class="sample-text">';
    html += '<span class="prompt">' + escHtml(promptPart) + '</span>';
    html += escHtml(genPart);
    html += '</div></div>';
  }

  html += '</div>';
  container.innerHTML = html;
}

// Known fixed prompts — highlight the prompt portion
const KNOWN_PROMPTS = {
  'recall_lovelace': 'Lovelace completed her paper. Sixty-six pages. ',
  'prediction_sequence': 'If one then two then three then ',
  'recall_reality': 'Reality is frequently ',
};

function findPromptBoundary(name, text) {
  if (KNOWN_PROMPTS[name]) {
    const p = KNOWN_PROMPTS[name];
    if (text.startsWith(p)) return p.length;
  }
  // For random samples, try to find first newline or sentence end
  const firstBreak = text.search(/[.!?\\n]/);
  if (firstBreak > 0 && firstBreak < 80) return firstBreak + 1;
  return Math.min(40, text.length);
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ================================================================
// KEYBOARD
// ================================================================

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') navEpoch(-1);
  if (e.key === 'ArrowRight') navEpoch(1);
  if (e.key === 'End') navEpoch(Infinity);
  if (e.key === 'Home') { viewEpochIdx = 0; render(); }
});
</script>
</body>
</html>"""


class MonitorHandler(http.server.BaseHTTPRequestHandler):
    """Serves the dashboard HTML and live data from the training log."""

    log_path = None  # Set by main()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        elif parsed.path == "/api/data":
            data = read_log(self.log_path)
            if data is None:
                self.send_response(404)
                self.end_headers()
                return

            payload = json.dumps(data).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass


def main():
    log_path = None
    port = DEFAULT_PORT

    if len(sys.argv) >= 2:
        log_path = sys.argv[1]
    if len(sys.argv) >= 3:
        port = int(sys.argv[2])

    if log_path is None:
        log_path = find_latest_log()
        if log_path is None:
            print(f"No JSON logs found in {LOG_DIR}/")
            print(f"Usage: python live_monitor.py [log_file.json] [port]")
            sys.exit(1)

    print(f"Monitoring: {log_path}")
    print(f"Dashboard:  http://localhost:{port}")
    print(f"Polling every 30s. Ctrl+C to stop.\n")

    # Verify file exists and is valid
    data = read_log(log_path)
    if data:
        epochs = data.get("epochs", [])
        cfg = data.get("config", {})
        print(f"  Model:  {cfg.get('model', '?')}")
        print(f"  Epochs: {len(epochs)}")
        if epochs:
            print(
                f"  Latest: epoch {epochs[-1]['epoch']}, val_loss={epochs[-1]['val_loss']:.4f}"
            )
    else:
        print(f"  (file empty or not yet created — will poll until data appears)")

    MonitorHandler.log_path = log_path

    server = http.server.HTTPServer(("127.0.0.1", port), MonitorHandler)

    # Open browser on a separate thread so we don't block
    threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
