"""
Puter.js Gemini Component — zero-API-key LLM via Puter.js.

Spins up a tiny background HTTP server that serves the Puter.js HTML page.
Uses st.components.v1.declare_component() for proper bidirectional communication
so Python actually receives the LLM response back from the iframe.
"""

import threading
import socket
import streamlit.components.v1 as components

# ── Module-level singletons ───────────────────────────────────────────────────
_server_port: int | None = None
_component_func = None
_lock = threading.Lock()

PUTER_MODEL = "gemini-3.1-flash-lite"

# ── Component HTML (served by background thread) ──────────────────────────────
_COMPONENT_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://js.puter.com/v2/"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Mono', monospace;
      font-size: 12px;
      background: transparent;
      padding: 8px 12px;
      color: #1de9b6;
    }
    #msg { display: flex; align-items: center; gap: 8px; }
    .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #1de9b6;
      animation: blink 1s infinite;
      flex-shrink: 0;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.1} }
  </style>
</head>
<body>
  <div id="msg"><span class="dot" id="dot"></span><span id="text">Initialising…</span></div>

<script>
// ── Minimal Streamlit component protocol ──────────────────────────────────────
const _Streamlit = {
  _ready: false,
  setComponentReady: function() {
    window.parent.postMessage({
      isStreamlitMessage: true,
      type: "streamlit:componentReady",
      apiVersion: 1
    }, "*");
  },
  setComponentValue: function(value) {
    window.parent.postMessage({
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: value,
      dataType: "json"
    }, "*");
  },
  setFrameHeight: function(h) {
    window.parent.postMessage({
      isStreamlitMessage: true,
      type: "streamlit:setFrameHeight",
      height: h
    }, "*");
  }
};

const dot  = document.getElementById('dot');
const text = document.getElementById('text');

function setStatus(msg, color) {
  text.textContent = msg;
  if (color) {
    text.style.color = color;
    dot.style.background = color;
    dot.style.animation = 'none';
  }
}

async function runChat(prompt) {
  setStatus('Calling Gemini via Puter.js — no API key needed…');
  try {
    const resp = await puter.ai.chat(prompt, { model: '%MODEL%' });

    let result = '';
    if (typeof resp === 'string') {
      result = resp;
    } else if (resp?.message?.content) {
      const c = resp.message.content;
      result = Array.isArray(c)
        ? c.map(x => x.text || '').join('')
        : String(c);
    } else if (resp?.text) {
      result = resp.text;
    } else {
      result = JSON.stringify(resp);
    }

    setStatus('✅ Report generated!', '#00e5ff');
    _Streamlit.setComponentValue({ ok: true, text: result });

  } catch (err) {
    const msg = err?.message || String(err);
    setStatus('❌ ' + msg, '#ff4b6e');
    _Streamlit.setComponentValue({ ok: false, error: msg, text: null });
  }
}

// ── Listen for Streamlit render event (sends component args) ──────────────────
window.addEventListener("message", function(event) {
  const d = event.data;
  if (!d || !d.isStreamlitMessage) return;

  if (d.type === "streamlit:render") {
    const prompt = d.args?.prompt || "";
    if (prompt) {
      runChat(prompt);
    } else {
      setStatus('⏳ Waiting for prompt…');
    }
  }
});

// ── Signal ready ──────────────────────────────────────────────────────────────
_Streamlit.setComponentReady();
_Streamlit.setFrameHeight(40);
</script>
</body>
</html>
""".replace("%MODEL%", PUTER_MODEL)


# ── Background HTTP server ────────────────────────────────────────────────────

class _Handler:
    """Minimal WSGI-free HTTP handler using http.server."""
    pass


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(port: int):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    html_bytes = _COMPONENT_HTML.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, *args):
            pass  # silence server logs

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()


def _ensure_server() -> int:
    """Start the background server once; return its port."""
    global _server_port
    with _lock:
        if _server_port is None:
            port = _find_free_port()
            t = threading.Thread(target=_serve, args=(port,), daemon=True)
            t.start()
            _server_port = port
    return _server_port


# ── Public component factory ──────────────────────────────────────────────────

def _get_component_func():
    """Return (and cache) the declare_component function."""
    global _component_func
    if _component_func is None:
        port = _ensure_server()
        _component_func = components.declare_component(
            "puter_gemini",
            url=f"http://127.0.0.1:{port}",
        )
    return _component_func


def puter_chat(prompt: str, key: str = "puter_chat") -> dict | None:
    """
    Render the Puter.js Gemini component and return the result dict.

    Returns:
        {"ok": True,  "text": "<llm response>"}   on success
        {"ok": False, "error": "<msg>", "text": None}  on failure
        None  while waiting (component hasn't responded yet)
    """
    fn = _get_component_func()
    result = fn(prompt=prompt, key=key, default=None)
    return result
