"""
VAJRA — Secure LLM Proxy Gateway
══════════════════════════════════════════════════════════════════════════════

Every request flows through this pipeline:

    CLIENT REQUEST
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L1 — Input Sanitization                                        │
    │  Unicode norm · homoglyph map · invisible-char strip            │
    │  repeated-char collapse · whitespace norm · pattern match       │
    └──────────────────────────────────┬──────────────────────────────┘
                                       │ clean_text
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L2 — Semantic Similarity + LLM Intent Classifier               │
    │  FAISS vector search (block/review/pass)                        │
    │  Gemini intent classifier (on REVIEW or FAISS disabled)         │
    └──────────────────────────────────┬──────────────────────────────┘
                                       │ l2 verdict
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L3 — Policy Enforcement + Tool Access Control                   │
    │  Rule engine · cross-layer escalation (L1+L2+L3 signals)        │
    │  Multi-turn conversation tracker · RBAC tool registry            │
    └──────────────────────────────────┬──────────────────────────────┘
                                       │ policy decision
                                       ▼
                            ┌────────────────────┐
                            │   GEMINI API CALL   │
                            └──────────┬─────────┘
                                       │ raw LLM response
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  L4 — Output PII + Secret + Safety Filtering                    │
    │  Presidio NER · Regex PII (SSN/card/IBAN/email/phone)           │
    │  Secret detection (AWS/JWT/RSA keys) · Content safety rules     │
    └──────────────────────────────────┬──────────────────────────────┘
                                       │ filtered_text
                                       ▼
                            CLIENT RECEIVES RESPONSE

ROUTES
──────
    POST /v1/chat/completions   Main proxy (OpenAI-compatible)
    GET  /health                Layer status + liveness
    GET  /metrics               Live stats (requests, blocks, latency)
    GET  /logs                  Audit log — last 500 requests (JSON)
    GET  /logs/stream           SSE live log stream for the dashboard
    GET  /layers                Active layer config summary
    POST /admin/reload          Hot-reload configs without restart
    GET  /docs                  Swagger UI (auto-generated)
    GET  /                      Live dashboard (HTML)

SETUP
──────
    pip install fastapi uvicorn httpx pyyaml python-dotenv
    echo "GEMINI_API_KEY=your_key_here" > .env
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    # Point your client at http://localhost:8000 instead of Gemini
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# ── dotenv (optional) ────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "vajra.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("vajra.main")

# ══════════════════════════════════════════════════════════════════════════════
#  IN-MEMORY AUDIT RING  (last 500 requests → /logs + /logs/stream SSE)
# ══════════════════════════════════════════════════════════════════════════════

_AUDIT: deque[dict] = deque(maxlen=500)
_SSE_QUEUES: list[asyncio.Queue] = []


def _audit_push(entry: dict) -> None:
    _AUDIT.appendleft(entry)
    dead = []
    for q in _SSE_QUEUES:
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _SSE_QUEUES.remove(q)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

class _Metrics:
    def __init__(self):
        self.total         = 0
        self.blocked       = 0
        self.passed        = 0
        self.blocks_by_layer: dict[str, int] = {}
        self.latencies: deque[float] = deque(maxlen=500)
        self._t0 = time.time()

    def record(self, blocked: bool, layer: Optional[str], ms: float):
        self.total += 1
        self.latencies.append(ms)
        if blocked and layer:
            self.blocked += 1
            self.blocks_by_layer[layer] = self.blocks_by_layer.get(layer, 0) + 1
        else:
            self.passed += 1

    def snapshot(self) -> dict:
        lats = sorted(self.latencies)
        n = len(lats)
        return {
            "uptime_s":        round(time.time() - self._t0, 1),
            "total":           self.total,
            "blocked":         self.blocked,
            "passed":          self.passed,
            "block_rate_pct":  round(100 * self.blocked / max(self.total, 1), 2),
            "blocks_by_layer": self.blocks_by_layer,
            "p50_ms":  round(lats[n // 2], 1)         if n else 0,
            "p95_ms":  round(lats[int(n * 0.95)], 1)  if n else 0,
            "p99_ms":  round(lats[int(n * 0.99)], 1)  if n else 0,
            "avg_ms":  round(sum(lats) / n, 1)        if n else 0,
        }


_metrics = _Metrics()

# ══════════════════════════════════════════════════════════════════════════════
#  LAYER REGISTRY  (graceful degradation — failed layer = warn + skip)
# ══════════════════════════════════════════════════════════════════════════════

class _LayerReg:
    def __init__(self):
        self.info: dict[str, dict] = {}

    def set(self, name: str, obj: Any, err: Optional[str] = None):
        self.info[name] = {
            "loaded": obj is not None,
            "class":  type(obj).__name__ if obj else None,
            "error":  err,
        }

    def summary(self) -> dict:
        return self.info


_reg = _LayerReg()


def _load(name: str, factory):
    """Try to instantiate a layer. Returns instance or None — never crashes."""
    try:
        obj = factory()
        _reg.set(name, obj)
        logger.info("✓ %-22s loaded", name)
        return obj
    except Exception as exc:
        _reg.set(name, None, err=str(exc))
        logger.warning("✗ %-22s FAILED (skipped) | %s", name, exc)
        return None


# ── Import each layer ─────────────────────────────────────────────────────────

layer1 = layer2 = layer3 = layer4 = None

try:
    from layers.layer1_sanitization import Layer1Sanitization
    layer1 = _load("L1_Sanitization", Layer1Sanitization)
except ImportError as e:
    _reg.set("L1_Sanitization", None, err=f"ImportError: {e}")
    logger.warning("L1 import failed: %s", e)

try:
    from layers.layer2_semantic import Layer2Semantic
    layer2 = _load("L2_Semantic", Layer2Semantic)
except ImportError as e:
    _reg.set("L2_Semantic", None, err=f"ImportError: {e}")
    logger.warning("L2 import failed: %s", e)

try:
    from layers.layer3_policy import Layer3Policy
    layer3 = _load("L3_Policy", Layer3Policy)
except ImportError as e:
    _reg.set("L3_Policy", None, err=f"ImportError: {e}")
    logger.warning("L3 import failed: %s", e)

try:
    from layers.layer4_output import Layer4Output
    layer4 = _load("L4_Output", Layer4Output)
except ImportError as e:
    _reg.set("L4_Output", None, err=f"ImportError: {e}")
    logger.warning("L4 import failed: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY  = "AIzaSyBXDHnpZgyqEmg35WEnWe_aOwZATP81Ga4"
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai",
)

if not GEMINI_API_KEY:
    logger.warning("⚠  GEMINI_API_KEY not set — upstream Gemini calls will fail")

# ══════════════════════════════════════════════════════════════════════════════
#  FASTAPI
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="VAJRA Secure LLM Proxy",
    description="4-layer security pipeline between your app and Gemini",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blocked_resp(layer: str, reason: str, rid: str, detail: dict = {}) -> dict:
    """OpenAI-shaped blocked response so the client never sees a raw error."""
    return {
        "id":      f"vajra-blocked-{rid}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   "vajra-proxy",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant",
                        "content": "⚠️ Request blocked by VAJRA security policy."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "vajra_metadata": {
            "blocked":    True,
            "blocked_by": layer,
            "reason":     reason,
            "request_id": rid,
            "detail":     detail,
            "timestamp":  _ts(),
        },
    }


def _ok_resp(llm_json: dict, rid: str, layers: dict) -> dict:
    """Attach VAJRA metadata to a passing Gemini response."""
    llm_json["vajra_metadata"] = {
        "blocked":      False,
        "request_id":   rid,
        "pii_redacted": layers.get("L4_Output", {}).get("redacted", False),
        "timestamp":    _ts(),
    }
    return llm_json


def _finalise(pipeline: dict, t0: float, blocked: bool, layer: Optional[str]):
    ms = round((time.perf_counter() - t0) * 1000, 2)
    pipeline["total_ms"]   = ms
    pipeline["blocked"]    = blocked
    pipeline["blocked_at"] = layer
    _audit_push(pipeline)
    _metrics.record(blocked, layer, ms)


# ══════════════════════════════════════════════════════════════════════════════
#  ① MAIN PROXY  — POST /v1/chat/completions
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/v1/chat/completions",
    tags=["Proxy"],
    summary="Main proxy — OpenAI-compatible chat completions",
)
async def proxy_chat(request: Request):
    """
    Drop-in replacement for the OpenAI / Gemini chat completions endpoint.

    **How to use as a proxy:**
    Point your existing OpenAI SDK or HTTP client at
    `http://localhost:8000` instead of `https://api.openai.com`.
    Every message flows through all loaded VAJRA layers before Gemini sees it.

    **Optional request headers:**
    | Header | Purpose |
    |---|---|
    | `X-Session-ID` | Multi-turn conversation tracking in L3 |
    | `X-User-Role` | RBAC role for tool access control in L3 |
    | `X-User-Identity` | Identity string for allow-list lookups in L3 |
    """
    t0  = time.perf_counter()
    rid = uuid.uuid4().hex[:12]

    # ── Parse body ───────────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    messages      = body.get("messages", [])
    user_msg      = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
    )
    session_id    = request.headers.get("X-Session-ID",    rid)
    user_role     = request.headers.get("X-User-Role",     "user")
    user_identity = request.headers.get("X-User-Identity", "")

    logger.info(
        "▶ [%s] request  model=%-25s  chars=%-5d  role=%s",
        rid, body.get("model", "?"), len(user_msg), user_role,
    )

    pipeline: dict[str, Any] = {
        "request_id":   rid,
        "timestamp":    _ts(),
        "user_message": user_msg[:400],
        "model":        body.get("model", ""),
        "session_id":   session_id,
        "user_role":    user_role,
        "blocked":      False,
        "blocked_at":   None,
        "layers":       {},
    }

    current = user_msg   # may be cleaned/replaced by L1

    # ════════════════════════════════════════════════════════════════════
    #  LAYER 1 — Input Sanitization
    # ════════════════════════════════════════════════════════════════════
    if layer1:
        t  = time.perf_counter()
        l1 = layer1.run(current, request_id=rid)
        l1["processing_ms"] = round((time.perf_counter() - t) * 1000, 2)
        pipeline["layers"]["L1_Sanitization"] = l1

        if l1["blocked"]:
            logger.warning("✗ [%s] BLOCKED L1 | reason=%s", rid, l1["reason"])
            _finalise(pipeline, t0, True, "L1_Sanitization")
            return JSONResponse(_blocked_resp("L1_Sanitization", l1["reason"], rid, l1))

        current = l1.get("clean_text", current)
        logger.info(
            "✓ [%s] L1 pass | transforms=%s  flags=%d  (%.0fms)",
            rid, l1.get("transforms", []), l1.get("flag_count", 0), l1["processing_ms"],
        )
    else:
        pipeline["layers"]["L1_Sanitization"] = {"skipped": True, "reason": "not loaded"}

    # ════════════════════════════════════════════════════════════════════
    #  LAYER 2 — Semantic + LLM Classifier
    # ════════════════════════════════════════════════════════════════════
    if layer2:
        t = time.perf_counter()
        try:
            l2 = await layer2.run(current, request_id=rid)
        except Exception as exc:
            # fail-open: log error, pass through
            logger.error("[%s] L2 error (fail-open): %s", rid, exc)
            l2 = {"blocked": False, "stage": "error", "error": str(exc)}
        l2["processing_ms"] = round((time.perf_counter() - t) * 1000, 2)
        pipeline["layers"]["L2_Semantic"] = l2

        if l2.get("blocked"):
            logger.warning(
                "✗ [%s] BLOCKED L2 | stage=%s  reason=%s",
                rid, l2.get("stage"), l2.get("reason"),
            )
            _finalise(pipeline, t0, True, "L2_Semantic")
            return JSONResponse(_blocked_resp("L2_Semantic", l2["reason"], rid, l2))

        logger.info(
            "✓ [%s] L2 pass | stage=%-6s  score=%.3f  (%.0fms)",
            rid, l2.get("stage", "?"),
            l2.get("faiss", {}).get("score", 0),
            l2["processing_ms"],
        )
    else:
        pipeline["layers"]["L2_Semantic"] = {"skipped": True, "reason": "not loaded"}

    # ════════════════════════════════════════════════════════════════════
    #  LAYER 3 — Policy + Tool Access
    # ════════════════════════════════════════════════════════════════════
    if layer3:
        t = time.perf_counter()
        try:
            l3 = layer3.run_text(
                text       = current,
                session_id = session_id,
                user_role  = user_role,
                identity   = user_identity,
                request_id = rid,
            )
        except Exception as exc:
            logger.error("[%s] L3 error (fail-open): %s", rid, exc)
            l3 = {"blocked": False, "error": str(exc), "action": "ALLOW"}
        l3["processing_ms"] = round((time.perf_counter() - t) * 1000, 2)
        pipeline["layers"]["L3_Policy"] = l3

        if l3.get("blocked"):
            logger.warning(
                "✗ [%s] BLOCKED L3 | severity=%-8s  reason=%s",
                rid, l3.get("severity"), l3.get("reason"),
            )
            _finalise(pipeline, t0, True, "L3_Policy")
            return JSONResponse(_blocked_resp("L3_Policy", l3["reason"], rid, l3))

        logger.info(
            "✓ [%s] L3 pass | action=%-5s  rules=%d  (%.0fms)",
            rid, l3.get("action", "?"),
            len(l3.get("triggered_rules", [])),
            l3["processing_ms"],
        )
    else:
        pipeline["layers"]["L3_Policy"] = {"skipped": True, "reason": "not loaded"}

    # ════════════════════════════════════════════════════════════════════
    #  FORWARD TO GEMINI
    # ════════════════════════════════════════════════════════════════════
    logger.info("→ [%s] forwarding to Gemini  model=%s", rid, body.get("model"))
    t_gem = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{GEMINI_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                    "Content-Type":  "application/json",
                },
                json=body,
            )
        gem = resp.json()
        pipeline["gemini_ms"] = round((time.perf_counter() - t_gem) * 1000, 2)

        if resp.status_code != 200:
            logger.error("[%s] Gemini HTTP %d: %s", rid, resp.status_code, gem)
            pipeline["gemini_error"] = gem
            _finalise(pipeline, t0, False, None)
            return JSONResponse(gem, status_code=resp.status_code)

        raw_text = (
            gem.get("choices", [{}])[0]
               .get("message", {})
               .get("content", "")
        )
        logger.info(
            "← [%s] Gemini OK  chars=%-5d  (%.0fms)",
            rid, len(raw_text), pipeline["gemini_ms"],
        )

    except httpx.TimeoutException:
        logger.error("[%s] Gemini timeout", rid)
        _finalise(pipeline, t0, False, None)
        return JSONResponse({"error": "Upstream Gemini timeout"}, status_code=504)
    except Exception as exc:
        logger.error("[%s] Gemini call failed: %s", rid, exc)
        _finalise(pipeline, t0, False, None)
        return JSONResponse({"error": str(exc)}, status_code=502)

    # ════════════════════════════════════════════════════════════════════
    #  LAYER 4 — Output Filtering
    # ════════════════════════════════════════════════════════════════════
    filtered = raw_text

    if layer4 and raw_text:
        t   = time.perf_counter()
        l4  = layer4.run(raw_text, request_id=rid)
        l4d = l4.to_dict()
        l4d["processing_ms"] = round((time.perf_counter() - t) * 1000, 2)
        pipeline["layers"]["L4_Output"] = l4d

        if l4.blocked:
            logger.warning(
                "✗ [%s] BLOCKED L4 | safety_hits=%d", rid, len(l4.safety_findings)
            )
            _finalise(pipeline, t0, True, "L4_Output")
            return JSONResponse(
                _blocked_resp("L4_Output",
                              "Output blocked by content safety filter", rid, l4d)
            )

        filtered = l4.filtered_text
        logger.info(
            "✓ [%s] L4 pass | redacted=%-5s  pii=%-2d  secrets=%-2d  (%.0fms)",
            rid, l4.redacted, len(l4.pii_findings),
            len(l4.secret_findings), l4d["processing_ms"],
        )
    else:
        pipeline["layers"]["L4_Output"] = {"skipped": True, "reason": "not loaded"}

    # ════════════════════════════════════════════════════════════════════
    #  SEND TO CLIENT
    # ════════════════════════════════════════════════════════════════════
    if gem.get("choices"):
        gem["choices"][0]["message"]["content"] = filtered

    pipeline["response_preview"] = filtered[:200]
    _finalise(pipeline, t0, False, None)

    logger.info("✓ [%s] DONE  total_ms=%.0f", rid, pipeline["total_ms"])
    return JSONResponse(_ok_resp(gem, rid, pipeline["layers"]))


# ══════════════════════════════════════════════════════════════════════════════
#  ② HEALTH  — GET /health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["Admin"], summary="Liveness check + per-layer status")
async def health():
    """
    Returns `status: running` plus details on every layer.

    **Quick test:**
    ```bash
    curl http://localhost:8000/health | python3 -m json.tool
    ```
    """
    s = _reg.summary()
    return {
        "status":         "running",
        "version":        "2.1.0",
        "gemini_key_set": bool(GEMINI_API_KEY),
        "layers_loaded":  [k for k, v in s.items() if v["loaded"]],
        "layers_missing": [k for k, v in s.items() if not v["loaded"]],
        "layer_detail":   s,
        "timestamp":      _ts(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ③ METRICS  — GET /metrics
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/metrics", tags=["Admin"], summary="Live counters and latency percentiles")
async def metrics():
    """
    Real-time performance snapshot.

    **Quick test:**
    ```bash
    curl http://localhost:8000/metrics | python3 -m json.tool
    ```
    """
    return _metrics.snapshot()


# ══════════════════════════════════════════════════════════════════════════════
#  ④ AUDIT LOG  — GET /logs
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/logs", tags=["Admin"], summary="Request audit log (newest first)")
async def get_logs(
    limit:        int  = Query(50,  ge=1, le=500, description="Max entries to return"),
    blocked_only: bool = Query(False, description="Return only blocked requests"),
    layer:        Optional[str] = Query(None,  description="Filter by blocked_at layer"),
):
    """
    Last N request audit entries from the in-memory ring buffer.

    **Quick tests:**
    ```bash
    # Last 10 requests
    curl "http://localhost:8000/logs?limit=10"

    # Only blocked requests
    curl "http://localhost:8000/logs?blocked_only=true"

    # Blocked by a specific layer
    curl "http://localhost:8000/logs?layer=L1_Sanitization"
    curl "http://localhost:8000/logs?layer=L3_Policy"
    ```
    """
    entries = list(_AUDIT)[:limit]
    if blocked_only:
        entries = [e for e in entries if e.get("blocked")]
    if layer:
        entries = [e for e in entries if e.get("blocked_at") == layer]
    return {"count": len(entries), "entries": entries}


# ══════════════════════════════════════════════════════════════════════════════
#  ⑤ SSE LIVE STREAM  — GET /logs/stream
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/logs/stream", tags=["Admin"], summary="Server-Sent Events live log stream")
async def stream_logs(request: Request):
    """
    Every request that hits the proxy is pushed here in real time.
    The dashboard connects to this endpoint automatically.

    **Test in terminal (stays open):**
    ```bash
    curl -N http://localhost:8000/logs/stream
    ```

    **Test in browser console:**
    ```js
    const es = new EventSource('http://localhost:8000/logs/stream');
    es.onmessage = e => console.log(JSON.parse(e.data));
    ```
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _SSE_QUEUES.append(q)

    async def gen() -> AsyncGenerator[str, None]:
        yield 'data: {"type":"connected"}\n\n'
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {json.dumps(entry, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            try:
                _SSE_QUEUES.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ⑥ LAYER INFO  — GET /layers
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/layers", tags=["Admin"], summary="Pipeline order + layer config summary")
async def layer_info():
    """
    Shows the pipeline in order and what loaded.

    **Quick test:**
    ```bash
    curl http://localhost:8000/layers | python3 -m json.tool
    ```
    """
    return {
        "pipeline": [
            "L1_Sanitization  →  unicode / homoglyph / invisible / pattern match",
            "L2_Semantic      →  FAISS similarity + Gemini intent classifier",
            "L3_Policy        →  rule engine + cross-layer escalation + multi-turn + RBAC",
            "[ Gemini API ]",
            "L4_Output        →  Presidio NER + regex PII + secret detection + safety",
        ],
        "layer_status": _reg.summary(),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  ① B  NATIVE GEMINI PROXY  — POST /v1beta/models/{model_path}
#  Catches calls from google-genai SDK / langchain-google-genai
# ══════════════════════════════════════════════════════════════════════════════

@app.api_route(
    "/v1beta/{full_path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
    tags=["Proxy"],
    summary="Native Gemini SDK catch-all proxy"
)
async def proxy_gemini_native(full_path: str, request: Request):
    # full_path will be: "models/gemini-2.5-flash:generateContent"
    t0  = time.perf_counter()
    rid = uuid.uuid4().hex[:12]

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Extract user text from native Gemini format
    user_msg = ""
    for c in reversed(body.get("contents", [])):
        if c.get("role") == "user":
            for part in c.get("parts", []):
                if "text" in part:
                    user_msg = part["text"]
                    break
        if user_msg:
            break

    session_id    = request.headers.get("X-Session-ID", rid)
    user_role     = request.headers.get("X-User-Role",  "user")
    user_identity = request.headers.get("X-User-Identity", "")

    logger.info("▶ [%s] native  /%s  chars=%d", rid, full_path, len(user_msg))

    pipeline: dict[str, Any] = {
        "request_id": rid, "timestamp": _ts(),
        "user_message": user_msg[:400], "model": full_path,
        "session_id": session_id, "user_role": user_role,
        "blocked": False, "blocked_at": None, "layers": {},
    }

    current = user_msg

    # ── L1 ──────────────────────────────────────────────────────────────────
    if layer1:
        t  = time.perf_counter()
        l1 = layer1.run(current, request_id=rid)
        l1["processing_ms"] = round((time.perf_counter() - t) * 1000, 2)
        pipeline["layers"]["L1_Sanitization"] = l1
        if l1["blocked"]:
            logger.warning("✗ [%s] BLOCKED L1 | %s", rid, l1["reason"])
            _finalise(pipeline, t0, True, "L1_Sanitization")
            return JSONResponse(
                {"error": {"code": 400, "message": f"Blocked by VAJRA L1: {l1['reason']}", "status": "INVALID_ARGUMENT"}},
                status_code=400,
            )
        current = l1.get("clean_text", current)
    else:
        pipeline["layers"]["L1_Sanitization"] = {"skipped": True}

    # ── L2 ──────────────────────────────────────────────────────────────────
    if layer2:
        try:
            l2 = await layer2.run(current, request_id=rid)
        except Exception as exc:
            l2 = {"blocked": False, "error": str(exc)}
        pipeline["layers"]["L2_Semantic"] = l2
        if l2.get("blocked"):
            logger.warning("✗ [%s] BLOCKED L2 | %s", rid, l2.get("reason"))
            _finalise(pipeline, t0, True, "L2_Semantic")
            return JSONResponse(
                {"error": {"code": 400, "message": f"Blocked by VAJRA L2: {l2.get('reason')}", "status": "INVALID_ARGUMENT"}},
                status_code=400,
            )
    else:
        pipeline["layers"]["L2_Semantic"] = {"skipped": True}

    # ── L3 ──────────────────────────────────────────────────────────────────
    if layer3:
        try:
            l3 = layer3.run_text(
                text=current, session_id=session_id,
                user_role=user_role, identity=user_identity, request_id=rid,
            )
        except Exception as exc:
            l3 = {"blocked": False, "error": str(exc), "action": "ALLOW"}
        pipeline["layers"]["L3_Policy"] = l3
        if l3.get("blocked"):
            logger.warning("✗ [%s] BLOCKED L3 | %s", rid, l3.get("reason"))
            _finalise(pipeline, t0, True, "L3_Policy")
            return JSONResponse(
                {"error": {"code": 400, "message": f"Blocked by VAJRA L3: {l3.get('reason')}", "status": "INVALID_ARGUMENT"}},
                status_code=400,
            )
    else:
        pipeline["layers"]["L3_Policy"] = {"skipped": True}

    # ── Forward to real Gemini ───────────────────────────────────────────────
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/{full_path}"
    logger.info("→ [%s] forwarding to %s", rid, gemini_url)
    t_gem = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                gemini_url,
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type":   "application/json",
                },
                json=body,
            )
        gem = resp.json()
        pipeline["gemini_ms"] = round((time.perf_counter() - t_gem) * 1000, 2)

        if resp.status_code != 200:
            logger.error("[%s] Gemini HTTP %d: %s", rid, resp.status_code, gem)
            _finalise(pipeline, t0, False, None)
            return JSONResponse(gem, status_code=resp.status_code)

        raw_text = ""
        try:
            raw_text = gem["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            pass

        logger.info("← [%s] Gemini OK  chars=%d  (%.0fms)", rid, len(raw_text), pipeline["gemini_ms"])

    except httpx.TimeoutException:
        _finalise(pipeline, t0, False, None)
        return JSONResponse({"error": "Upstream timeout"}, status_code=504)
    except Exception as exc:
        logger.error("[%s] Gemini call failed: %s", rid, exc)
        _finalise(pipeline, t0, False, None)
        return JSONResponse({"error": str(exc)}, status_code=502)

    # ── L4 ──────────────────────────────────────────────────────────────────
    if layer4 and raw_text:
        l4  = layer4.run(raw_text, request_id=rid)
        l4d = l4.to_dict()
        pipeline["layers"]["L4_Output"] = l4d
        if l4.blocked:
            logger.warning("✗ [%s] BLOCKED L4", rid)
            _finalise(pipeline, t0, True, "L4_Output")
            return JSONResponse(
                {"error": {"code": 400, "message": "Output blocked by VAJRA L4", "status": "INVALID_ARGUMENT"}},
                status_code=400,
            )
        try:
            gem["candidates"][0]["content"]["parts"][0]["text"] = l4.filtered_text
        except (KeyError, IndexError):
            pass
    else:
        pipeline["layers"]["L4_Output"] = {"skipped": True}

    pipeline["response_preview"] = raw_text[:200]
    _finalise(pipeline, t0, False, None)
    logger.info("✓ [%s] native DONE  total_ms=%.0f", rid, pipeline["total_ms"])
    return JSONResponse(gem)

# ══════════════════════════════════════════════════════════════════════════════
#  ⑧ LIVE DASHBOARD  — GET /
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    return HTMLResponse(_DASHBOARD)


_DASHBOARD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAJRA — Secure LLM Proxy</title>
<style>
:root{
  --bg:#0d1117;--panel:#161b22;--border:#21262d;
  --green:#3fb950;--red:#f85149;--yellow:#d29922;
  --blue:#58a6ff;--purple:#bc8cff;--orange:#ffa657;
  --text:#e6edf3;--muted:#8b949e;
  --font:'JetBrains Mono','Fira Code',monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;line-height:1.6}

/* header */
header{padding:14px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px}
header h1{font-size:17px;letter-spacing:3px;color:var(--blue);font-weight:700}
.tag{padding:2px 9px;border-radius:20px;font-size:11px;border:1px solid var(--border)}
.tag.live{border-color:var(--green);color:var(--green)}
.tag.live::before{content:"● ";animation:blink 1.4s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
#clock{margin-left:auto;color:var(--muted);font-size:11px}

/* stat grid */
.stat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;padding:16px 24px}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:14px 16px}
.stat .lbl{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted)}
.stat .val{font-size:24px;font-weight:700;margin-top:4px}
.c-green{color:var(--green)}.c-red{color:var(--red)}.c-blue{color:var(--blue)}.c-yellow{color:var(--yellow)}

/* sections */
.sec{padding:0 24px 18px}
.sec-hdr{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);
          padding-bottom:8px;border-bottom:1px solid var(--border);margin-bottom:10px}

/* layer pills */
.layer-row{display:flex;flex-wrap:wrap;gap:8px}
.lpill{display:flex;align-items:center;gap:8px;background:var(--panel);
       border:1px solid var(--border);border-radius:6px;padding:8px 14px;min-width:230px}
.dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.dot.ok{background:var(--green);box-shadow:0 0 5px var(--green)}
.dot.err{background:var(--red);box-shadow:0 0 5px var(--red)}
.dot.skip{background:var(--yellow)}
.lname{font-weight:600;font-size:12px}
.lsub{color:var(--muted);font-size:10px;margin-top:1px}

/* bar chart */
.bar-row{display:flex;align-items:center;gap:8px;margin-bottom:6px}
.bar-lbl{width:160px;color:var(--muted);font-size:11px;text-align:right}
.bar-track{flex:1;background:var(--border);border-radius:3px;height:7px}
.bar-fill{height:100%;border-radius:3px;background:var(--red);transition:width .4s}
.bar-cnt{width:30px;font-size:11px}

/* log */
#log-wrap{background:var(--panel);border:1px solid var(--border);border-radius:8px;height:380px;overflow-y:auto}
.log-hdr,.log-row{display:grid;grid-template-columns:158px 110px 130px 1fr;gap:8px;padding:7px 14px;border-bottom:1px solid var(--border)}
.log-hdr{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:1px;position:sticky;top:0;background:var(--panel)}
.log-row:hover{background:rgba(255,255,255,.025);cursor:pointer}
.log-row .ts{color:var(--muted);font-size:11px}
.log-row .rid{color:var(--purple);font-size:11px}
.bp{color:var(--green);border:1px solid var(--green);background:rgba(63,185,80,.1);padding:1px 7px;border-radius:3px;font-size:10px;display:inline-block}
.bb{color:var(--red);border:1px solid var(--red);background:rgba(248,81,73,.1);padding:1px 7px;border-radius:3px;font-size:10px;display:inline-block}
.log-row .msg{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* drawer */
#drawer{position:fixed;top:0;right:-480px;width:480px;height:100vh;background:var(--panel);
        border-left:1px solid var(--border);overflow-y:auto;transition:right .22s;padding:20px;z-index:200}
#drawer.open{right:0}
#drawer h3{color:var(--blue);margin-bottom:12px}
#drawer pre{font-size:11px;white-space:pre-wrap;color:var(--muted);line-height:1.75}
.xbtn{float:right;cursor:pointer;color:var(--red);font-size:20px;line-height:1}

footer{padding:12px 24px;color:var(--muted);font-size:11px;border-top:1px solid var(--border)}
footer a{color:var(--blue);text-decoration:none}
</style>
</head>
<body>

<header>
  <h1>⚡ VAJRA</h1>
  <span class="tag">Secure LLM Proxy</span>
  <span class="tag live" id="conn">CONNECTING</span>
  <span id="clock"></span>
</header>

<div class="stat-grid">
  <div class="stat"><div class="lbl">Total Requests</div><div class="val c-blue"   id="s-total">—</div></div>
  <div class="stat"><div class="lbl">Blocked</div>        <div class="val c-red"    id="s-blocked">—</div></div>
  <div class="stat"><div class="lbl">Passed</div>         <div class="val c-green"  id="s-passed">—</div></div>
  <div class="stat"><div class="lbl">Block Rate</div>     <div class="val c-yellow" id="s-rate">—</div></div>
  <div class="stat"><div class="lbl">p50 Latency</div>   <div class="val c-blue"   id="s-p50">—</div></div>
  <div class="stat"><div class="lbl">p95 Latency</div>   <div class="val c-blue"   id="s-p95">—</div></div>
  <div class="stat"><div class="lbl">Uptime</div>         <div class="val c-green"  id="s-up">—</div></div>
</div>

<div class="sec">
  <div class="sec-hdr">Layer Status</div>
  <div class="layer-row" id="layers">
    <div class="lpill"><div class="dot skip"></div><div><div class="lname">Loading…</div></div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-hdr">Blocks by Layer</div>
  <div id="bars"><div style="color:var(--muted);padding:4px 0">No blocks recorded yet</div></div>
</div>

<div class="sec">
  <div class="sec-hdr">Live Request Log &nbsp;
    <span style="font-weight:400;text-transform:none;letter-spacing:0">— click any row for full pipeline detail</span>
  </div>
  <div id="log-wrap">
    <div class="log-hdr">
      <span>Timestamp (UTC)</span><span>Request ID</span><span>Status</span><span>User Message</span>
    </div>
  </div>
</div>

<div id="drawer">
  <span class="xbtn" onclick="closeDrawer()">✕</span>
  <h3>Pipeline Detail</h3>
  <pre id="drawer-body"></pre>
</div>

<footer>
  VAJRA v2.1 &nbsp;·&nbsp;
  <a href="/docs">Swagger UI</a> &nbsp;·&nbsp;
  <a href="/redoc">ReDoc</a> &nbsp;·&nbsp;
  <a href="/metrics">Metrics JSON</a> &nbsp;·&nbsp;
  <a href="/health">Health</a> &nbsp;·&nbsp;
  <a href="/logs">Logs JSON</a>
</footer>

<script>
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/</g,'&lt;').replace(/>/g,'&gt;');
const fmtMs=ms=>ms>=1000?(ms/1000).toFixed(1)+'s':ms+'ms';
const fmtUp=s=>{const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=Math.floor(s%60);return h?`${h}h ${m}m`:m?`${m}m ${ss}s`:`${ss}s`;};

// clock
setInterval(()=>{ $('clock').textContent=new Date().toISOString().replace('T',' ').slice(0,19)+' UTC'; },1000);

// metrics
async function pollMetrics(){
  try{
    const m=await fetch('/metrics').then(r=>r.json());
    $('s-total').textContent   = m.total;
    $('s-blocked').textContent = m.blocked;
    $('s-passed').textContent  = m.passed;
    $('s-rate').textContent    = m.block_rate_pct+'%';
    $('s-p50').textContent     = fmtMs(m.p50_ms);
    $('s-p95').textContent     = fmtMs(m.p95_ms);
    $('s-up').textContent      = fmtUp(m.uptime_s);

    const bl=m.blocks_by_layer||{};
    const maxV=Math.max(1,...Object.values(bl));
    $('bars').innerHTML=Object.keys(bl).length
      ? Object.entries(bl).map(([k,v])=>`
          <div class="bar-row">
            <div class="bar-lbl">${esc(k)}</div>
            <div class="bar-track"><div class="bar-fill" style="width:${(v/maxV*100).toFixed(1)}%"></div></div>
            <div class="bar-cnt">${v}</div>
          </div>`).join('')
      : '<div style="color:var(--muted);padding:4px 0">No blocks recorded yet</div>';
  }catch(_){}
}

// layers
async function pollLayers(){
  try{
    const d=await fetch('/layers').then(r=>r.json());
    const st=d.layer_status||{};
    $('layers').innerHTML=Object.entries(st).map(([k,v])=>`
      <div class="lpill">
        <div class="dot ${v.loaded?'ok':'err'}"></div>
        <div>
          <div class="lname">${esc(k)}</div>
          <div class="lsub">${v.loaded?(v.class||'loaded'):('✗ '+(v.error||'').slice(0,50))}</div>
        </div>
      </div>`).join('');
  }catch(_){}
}

pollMetrics(); pollLayers();
setInterval(pollMetrics,4000);
setInterval(pollLayers,20000);

// log rows
let rowCount=0;
function addRow(e){
  if(e.type==='connected') return;
  const blocked=!!e.blocked;
  const ts=(e.timestamp||'').slice(0,19).replace('T',' ');
  const rid=(e.request_id||'').slice(0,10);
  const blAt=e.blocked_at?` → ${esc(e.blocked_at)}`:'';
  const msg=esc((e.user_message||'(no message)').slice(0,130));
  const div=document.createElement('div');
  div.className='log-row';
  div.innerHTML=`
    <span class="ts">${ts}</span>
    <span class="rid">${rid}</span>
    <span class="${blocked?'bb':'bp'}">${blocked?'BLOCKED'+blAt:'PASSED'}</span>
    <span class="msg">${msg}</span>`;
  div.onclick=()=>openDrawer(e);
  const wrap=$('log-wrap');
  wrap.insertBefore(div, wrap.children[1]);
  if(++rowCount>300) wrap.lastChild.remove();
}

function openDrawer(e){ $('drawer-body').textContent=JSON.stringify(e,null,2); $('drawer').classList.add('open'); }
function closeDrawer(){ $('drawer').classList.remove('open'); }
document.addEventListener('click',ev=>{ if(!$('drawer').contains(ev.target)&&!ev.target.closest('.log-row')) closeDrawer(); });

// SSE
function connectSSE(){
  const es=new EventSource('/logs/stream');
  es.onopen=()=>{ $('conn').textContent='LIVE'; };
  es.onmessage=ev=>{ try{ addRow(JSON.parse(ev.data)); }catch(_){} };
  es.onerror=()=>{ $('conn').textContent='RECONNECTING'; es.close(); setTimeout(connectSSE,3000); };
}
connectSSE();
</script>
</body>
</html>
"""