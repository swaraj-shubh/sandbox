"""
VAJRA — Layer 2: Semantic Similarity + LLM Intent Classification
═══════════════════════════════════════════════════════════════════════════════
Pipeline (runs in sequence — earlier stages short-circuit to save cost/latency):

  2a. FAISS Semantic Search  (local, ~2-5ms after warmup, zero API cost)
      ├─ Score >= threshold_block  → BLOCK immediately, skip LLM
      ├─ Score >= threshold_review → REVIEW, escalate to LLM classifier
      └─ Score <  threshold_review → PASS, skip LLM

  2b. LLM Intent Classifier  (Gemini, ~300-800ms, only called on REVIEW or if FAISS disabled)
      ├─ category == "not_safe"    → BLOCK with reason + confidence
      └─ category == "safe"        → PASS to Layer 3

Design principles:
  - FAISS catches known/similar attacks cheaply — protects Gemini API budget
  - LLM catches creative/novel bypasses that regex + embedding both miss
  - fail_open: on API failure, pass through (availability > security at the edge)
  - All blocking decisions are logged with reason + confidence + duration
  - Hot-reload: call reload_config() to pick up config changes without restart

References:
  - OWASP LLM01: Prompt Injection
  - Codemancers (2026) "Protecting Your LLM Applications from Prompt Injection"
  - Pangea (2024) 300K attack dataset — LLM-driven analysis reduces bypass to 0.003%
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import logging
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from dotenv import load_dotenv, find_dotenv


# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv(find_dotenv(usecwd=True))
GEMINI_API_KEY = "AIzaSyCwZpSnRXEZzXUuK18JtgqRHCOfqWe8yKI"


# ── Paths ─────────────────────────────────────────────────────────────────────

PROXY_DIR   = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROXY_DIR / "config" / "layer2_config.yaml"
LOG_DIR     = PROXY_DIR / "logs"


# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("vajra.layer2")


def _setup_logger():
    LOG_DIR.mkdir(exist_ok=True)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return
    fh = logging.FileHandler(LOG_DIR / "vajra_layer2.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    ))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[VAJRA L2] %(levelname)s — %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)


_setup_logger()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Strip ```json ... ``` fences from LLM response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _result(blocked: bool, reason: Optional[str], stage: str,
            faiss: dict, llm: dict, duration_ms: float) -> dict:
    return {
        "blocked":     blocked,
        "reason":      reason,
        "stage":       stage,          # "faiss" | "llm" | "pass" | "empty"
        "faiss":       faiss,
        "llm":         llm,
        "layer":       "L2_Semantic",
        "duration_ms": round(duration_ms, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2
# ═════════════════════════════════════════════════════════════════════════════

class Layer2Semantic:
    """
    Layer 2 — FAISS semantic search + Gemini intent classification.

    FAISS acts as a cheap pre-filter. If it's confident (>= threshold_block)
    the request is rejected immediately — no API call, no cost.
    If uncertain (>= threshold_review) or FAISS is disabled, the LLM
    classifier makes the final call.

    Both can be independently enabled/disabled in layer2_config.yaml.
    """

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self):
        # Config
        self.faiss_enabled    = True
        self.threshold_block  = 0.82
        self.threshold_review = 0.65
        self.llm_enabled      = True
        self.llm_model        = "gemini-2.5-flash"
        self.llm_timeout      = 10
        self.fail_open        = True
        self.system_prompt    = ""
        self.seed_attacks:  list[str] = []

        # Runtime objects (set during init)
        self.embedding_model  = None
        self.faiss_index      = None
        self.attack_labels:   list[str] = []
        self.llm_client       = None

        # Load everything
        self._load_config()
        self._init_embeddings()
        self._init_faiss()
        self._init_llm()

    # ── Config ────────────────────────────────────────────────────────────────

    def _load_config(self):
        if not CONFIG_PATH.exists():
            logger.error(f"Config not found: {CONFIG_PATH}")
            return

        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        self.seed_attacks = cfg.get("seed_attacks", [])

        fc = cfg.get("faiss", {})
        self.faiss_enabled    = fc.get("enabled", True)
        self.threshold_block  = float(fc.get("threshold_block",  0.82))
        self.threshold_review = float(fc.get("threshold_review", 0.65))

        lc = cfg.get("llm_classifier", {})
        self.llm_enabled   = lc.get("enabled", True)
        self.llm_model     = lc.get("model", "gemini-2.5-flash")
        self.llm_timeout   = int(lc.get("timeout_seconds", 10))
        self.fail_open     = lc.get("fail_open", True)
        self.system_prompt = lc.get("system_prompt", "").strip()

        logger.info(
            f"Config loaded | "
            f"FAISS={'on' if self.faiss_enabled else 'off'} "
            f"block={self.threshold_block} review={self.threshold_review} | "
            f"LLM={'on' if self.llm_enabled else 'off'} model={self.llm_model} | "
            f"seeds={len(self.seed_attacks)} | fail_open={self.fail_open}"
        )

    def reload_config(self):
        """Hot-reload config + rebuild FAISS index. Call without restarting server."""
        logger.info("Hot-reloading Layer 2 config...")
        self._load_config()
        self._init_faiss()   # rebuild index with new seeds/thresholds

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _init_embeddings(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("Loading sentence-transformers/BAAI/bge-small-en-v1.5 ...")
            self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model ready")
        except ImportError:
            logger.warning(
                "langchain-huggingface not installed — FAISS disabled.\n"
                "  pip install langchain-huggingface sentence-transformers faiss-cpu"
            )
            self.embedding_model = None
            self.faiss_enabled = False
        except Exception as e:
            logger.error(f"Embedding init failed: {e}")
            self.embedding_model = None
            self.faiss_enabled = False

    # ── FAISS ─────────────────────────────────────────────────────────────────

    def _init_faiss(self):
        if not self.faiss_enabled:
            return
        if not self.embedding_model:
            logger.warning("Embeddings unavailable — FAISS skipped")
            return
        if not self.seed_attacks:
            logger.warning("No seed attacks in config — FAISS index empty, disabling")
            self.faiss_enabled = False
            return

        try:
            import faiss as faiss_lib

            logger.info(f"Building FAISS index from {len(self.seed_attacks)} seed attacks...")
            vectors = np.array(
                self.embedding_model.embed_documents(self.seed_attacks),
                dtype="float32"
            )

            # normalize_embeddings=True in HuggingFaceEmbeddings means vectors are
            # already unit-norm, but we normalize again defensively for IndexFlatIP
            faiss_lib.normalize_L2(vectors)

            dim = vectors.shape[1]
            self.faiss_index = faiss_lib.IndexFlatIP(dim)   # inner product = cosine on unit vectors
            self.faiss_index.add(vectors)
            self.attack_labels = list(self.seed_attacks)

            logger.info(
                f"FAISS ready | dim={dim} | vectors={self.faiss_index.ntotal} | "
                f"block>={self.threshold_block} review>={self.threshold_review}"
            )

        except ImportError:
            logger.warning("faiss-cpu not installed — FAISS disabled. pip install faiss-cpu")
            self.faiss_enabled = False
            self.faiss_index = None
        except Exception as e:
            logger.error(f"FAISS build failed: {e}")
            self.faiss_enabled = False
            self.faiss_index = None

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _init_llm(self):
        if not self.llm_enabled:
            return
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not set — LLM classifier will be skipped")
            return
        try:
            from google import genai
            self.llm_client = genai.Client(api_key=GEMINI_API_KEY)
            logger.info(f"Gemini client ready | model={self.llm_model}")
        except ImportError:
            logger.warning("google-generativeai not installed. pip install google-generativeai")
            self.llm_client = None
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")
            self.llm_client = None

    # ── Sub-check 2a: FAISS ───────────────────────────────────────────────────

    def _run_faiss(self, text: str) -> dict:
        """
        Embed text and search against known attack vectors.
        Returns: {score, action: BLOCK|REVIEW|PASS, closest_match, top3}
        """
        try:
            import faiss as faiss_lib

            vec = np.array(
                [self.embedding_model.embed_query(text)],
                dtype="float32"
            )
            faiss_lib.normalize_L2(vec)

            k = min(3, len(self.attack_labels))
            scores, indices = self.faiss_index.search(vec, k)

            top_score = float(scores[0][0])
            top_match = self.attack_labels[int(indices[0][0])]

            top3 = [
                {"score": round(float(scores[0][i]), 4),
                 "match": self.attack_labels[int(indices[0][i])]}
                for i in range(k)
            ]

            action = (
                "BLOCK"  if top_score >= self.threshold_block  else
                "REVIEW" if top_score >= self.threshold_review else
                "PASS"
            )

            logger.debug(
                f"FAISS | score={top_score:.4f} | action={action} | "
                f"match='{top_match[:50]}'"
            )

            return {
                "ran": True,
                "score": round(top_score, 4),
                "action": action,
                "closest_match": top_match,
                "top3": top3,
            }

        except Exception as e:
            logger.error(f"FAISS inference error: {e}")
            return {"ran": True, "action": "REVIEW", "score": 0.0, "error": str(e)}

    # ── Sub-check 2b: LLM Classifier ─────────────────────────────────────────

    async def _run_llm_classifier(self, text: str, request_id: str) -> dict:
        """
        Send text to Gemini for intent classification.
        Returns: {ran, category: safe|not_safe, reason, confidence, duration_ms}

        Uses asyncio.to_thread to keep the FastAPI event loop unblocked.
        fail_open=True means a timeout or API error → safe (availability wins).
        """
        if not self.llm_client:
            return {"ran": False, "category": "safe", "reason": "no_llm_client"}

        # Build prompt — system_prompt first, then structured instruction, then input
        prompt = (
            f"{self.system_prompt}\n\n"
            "━━━ INPUT TO CLASSIFY ━━━\n"
            f"{text}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Respond with ONLY this JSON (no markdown, no explanation):\n"
            '{"category":"safe|not_safe","reason":"one sentence","confidence":0.0}'
        )

        t = time.perf_counter()

        try:
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.llm_client.models.generate_content,
                    model=self.llm_model,
                    contents=prompt,
                ),
                timeout=self.llm_timeout,
            )

            duration_ms = round((time.perf_counter() - t) * 1000, 1)
            raw_text    = getattr(raw_response, "text", "") or ""
            parsed      = json.loads(_strip_fences(raw_text))

            category   = str(parsed.get("category", "safe")).lower().strip()
            reason     = str(parsed.get("reason",   "no reason given"))
            confidence = float(parsed.get("confidence", 0.5))

            # Normalise — Gemini sometimes returns "not safe" with a space
            if "not" in category and "safe" in category:
                category = "not_safe"
            elif category not in ("safe", "not_safe"):
                category = "safe"

            logger.debug(
                f"[{request_id}] LLM | cat={category} "
                f"conf={confidence:.0%} reason='{reason}' {duration_ms}ms"
            )

            return {
                "ran":         True,
                "category":    category,
                "reason":      reason,
                "confidence":  confidence,
                "duration_ms": duration_ms,
            }

        except asyncio.TimeoutError:
            duration_ms = round((time.perf_counter() - t) * 1000, 1)
            logger.warning(
                f"[{request_id}] LLM timeout after {self.llm_timeout}s "
                f"| fail_open={self.fail_open} | {duration_ms}ms"
            )
            return {
                "ran":        True,
                "category":   "safe" if self.fail_open else "not_safe",
                "reason":     f"classifier_timeout (fail_open={self.fail_open})",
                "confidence": 0.0,
                "timed_out":  True,
                "duration_ms": duration_ms,
            }

        except json.JSONDecodeError as e:
            raw_preview = getattr(raw_response, "text", "")[:120] if "raw_response" in dir() else ""
            logger.warning(
                f"[{request_id}] LLM non-JSON response: '{raw_preview}' | {e}"
            )
            return {
                "ran":         True,
                "category":    "safe",
                "reason":      "json_parse_error",
                "confidence":  0.0,
                "raw_preview": raw_preview,
            }

        except Exception as e:
            duration_ms = round((time.perf_counter() - t) * 1000, 1)
            logger.error(f"[{request_id}] LLM error: {type(e).__name__}: {e}")
            return {
                "ran":        True,
                "category":   "safe" if self.fail_open else "not_safe",
                "reason":     f"{type(e).__name__}: {e}",
                "confidence": 0.0,
                "duration_ms": duration_ms,
            }

    # ── Main entry ────────────────────────────────────────────────────────────

    async def run(self, text: str, request_id: str = "req") -> dict:
        """
        Run Layer 2 checks on sanitized text (output of Layer 1).

        Flow:
          ┌─ empty input          → pass (L1 handles this but we guard anyway)
          ├─ FAISS BLOCK          → return blocked immediately (no LLM call)
          ├─ FAISS PASS           → return not-blocked immediately (no LLM call)
          ├─ FAISS REVIEW         → run LLM classifier
          ├─ FAISS disabled       → run LLM classifier
          └─ LLM not_safe         → return blocked with reason + confidence

        Returns dict with keys:
          blocked, reason, stage, faiss, llm, layer, duration_ms
        """
        t_start      = time.perf_counter()
        faiss_result = {"ran": False, "action": "SKIP"}
        llm_result   = {"ran": False}

        # ── Guard: empty input ────────────────────────────────────────────────
        if not text or not text.strip():
            return _result(False, None, "empty", faiss_result, llm_result, 0.0)

        # ── 2a: FAISS semantic search ─────────────────────────────────────────
        if self.faiss_enabled and self.faiss_index is not None:

            faiss_result = self._run_faiss(text)
            action       = faiss_result.get("action", "PASS")
            score        = faiss_result.get("score",  0.0)

            if action == "BLOCK":
                ms     = (time.perf_counter() - t_start) * 1000
                match  = faiss_result.get("closest_match", "")[:60]
                reason = (
                    f"Semantic similarity {score:.3f} >= {self.threshold_block} | "
                    f"closest: '{match}'"
                )
                logger.warning(
                    f"[{request_id}] *** L2a FAISS BLOCKED *** | "
                    f"score={score:.4f} | '{text[:80]}' | {ms:.1f}ms"
                )
                return _result(True, reason, "faiss", faiss_result, llm_result, ms)

            if action == "PASS":
                ms = (time.perf_counter() - t_start) * 1000
                logger.debug(
                    f"[{request_id}] L2a FAISS PASS | score={score:.4f} | {ms:.1f}ms"
                )
                return _result(False, None, "pass", faiss_result, llm_result, ms)

            # REVIEW — fall through to LLM
            logger.info(
                f"[{request_id}] L2a FAISS REVIEW | score={score:.4f} → LLM classifier"
            )

        # ── 2b: LLM intent classifier ─────────────────────────────────────────
        if self.llm_enabled:

            llm_result = await self._run_llm_classifier(text, request_id)
            ms         = (time.perf_counter() - t_start) * 1000
            category   = llm_result.get("category",   "safe")
            confidence = llm_result.get("confidence",  0.0)
            llm_reason = llm_result.get("reason",      "")

            if category == "not_safe":
                reason = (
                    f"LLM: NOT_SAFE | confidence={confidence:.0%} | {llm_reason}"
                )
                logger.warning(
                    f"[{request_id}] *** L2b LLM BLOCKED *** | "
                    f"conf={confidence:.0%} | reason='{llm_reason}' | "
                    f"preview='{text[:80]}' | {ms:.1f}ms"
                )
                return _result(True, reason, "llm", faiss_result, llm_result, ms)

            logger.info(
                f"[{request_id}] L2b LLM PASS | "
                f"cat={category} conf={confidence:.0%} | {ms:.1f}ms"
            )
            return _result(False, None, "pass", faiss_result, llm_result, ms)

        # ── Both checks disabled ──────────────────────────────────────────────
        ms = (time.perf_counter() - t_start) * 1000
        logger.warning(
            f"[{request_id}] L2: FAISS and LLM both disabled — passing through. "
            "Enable at least one in config/layer2_config.yaml"
        )
        return _result(False, "both_checks_disabled", "pass", faiss_result, llm_result, ms)