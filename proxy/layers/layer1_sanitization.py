"""
VAJRA — Layer 1: Input Sanitization & Validation
═══════════════════════════════════════════════════════════════════════════════
What this layer does:
  1. Unicode normalization     → defeats homoglyph attacks  (ＩＧＮＯＲＥ → IGNORE)
  2. Cyrillic homoglyph map    → explicit Cyrillic/Greek→Latin map
                                  NFKC alone does NOT do this — they are different
                                  Unicode chars. і→i, о→o, а→a, р→p, у→y, etc.
  3. Invisible char stripping  → defeats ZWJ + RTLO (U+202E) obfuscation
  4. Repeated char collapse    → defeats elongation ("iiiignore" → "ignore")
                                  collapses 3+ repeated chars to 1 (not 2)
  5. Whitespace normalization  → defeats space-padding tricks
  6. Pattern matching          → regex against YAML-configured attack patterns
  7. Severity-based blocking   → CRITICAL/HIGH block immediately, MEDIUM needs 2+

Fixes vs original:
  [FIX-1] Added _apply_homoglyph_map — NFKC doesn't map Cyrillic to Latin
  [FIX-2] Added RTLO U+202E and all bidi control chars to invisible strip set
  [FIX-3] _collapse_repeated_chars now collapses 3+ → 1 (was 3+ → 2)
          'iiiignore' → 'ignore'  (was 'iignore' which didn't match any pattern)
  [FIX-4] _sanitize calls homoglyph map after NFKC normalization

Research basis:
  - OWASP LLM01: Prompt Injection
  - Perez & Ribeiro (2022) "Ignore Previous Prompt"
  - Greshake et al. (2023) indirect injection via documents
  - Pangea 300K attack dataset findings (2024)
═══════════════════════════════════════════════════════════════════════════════
"""

import re
import unicodedata
import logging
import time
from pathlib import Path
from typing import Optional
import yaml


# ── Module-level logger ───────────────────────────────────────────────────────
logger = logging.getLogger("vajra.layer1")


def _setup_logger():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_dir / "vajra_layer1.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    ))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="[VAJRA L1] %(levelname)s — %(message)s"))

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)


_setup_logger()

SEVERITY_RANK = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

BASE_DIR = Path(__file__).resolve().parent.parent


# ── [FIX-1] Cyrillic + Greek → Latin homoglyph table ─────────────────────────
# NFKC normalization does NOT convert Cyrillic to Latin because they are valid
# distinct Unicode characters. This map handles the visual lookalikes used in
# homoglyph attacks: 'іgnore' (Cyrillic і) looks identical to 'ignore'.
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic lowercase
    "\u0430": "a",   # а → a
    "\u0441": "c",   # с → c
    "\u0435": "e",   # е → e
    "\u0451": "e",   # ё → e
    "\u0456": "i",   # і → i  (Ukrainian і — most common attack char)
    "\u0406": "I",   # І → I
    "\u043a": "k",   # к → k
    "\u043d": "n",   # н → n
    "\u043e": "o",   # о → o
    "\u0440": "p",   # р → p
    "\u0442": "t",   # т → t
    "\u0443": "y",   # у → y
    "\u0445": "x",   # х → x
    # Cyrillic uppercase
    "\u0410": "A",   # А → A
    "\u0412": "B",   # В → B
    "\u0421": "C",   # С → C
    "\u0415": "E",   # Е → E
    "\u0401": "E",   # Ё → E
    "\u041a": "K",   # К → K
    "\u041c": "M",   # М → M
    "\u041d": "H",   # Н → H
    "\u041e": "O",   # О → O
    "\u0420": "P",   # Р → P
    "\u0422": "T",   # Т → T
    "\u0423": "Y",   # У → Y
    "\u0425": "X",   # Х → X
    # Greek uppercase lookalikes
    "\u0391": "A",   # Α → A
    "\u0392": "B",   # Β → B
    "\u0395": "E",   # Ε → E
    "\u0397": "H",   # Η → H
    "\u0399": "I",   # Ι → I
    "\u039a": "K",   # Κ → K
    "\u039c": "M",   # Μ → M
    "\u039d": "N",   # Ν → N
    "\u039f": "O",   # Ο → O
    "\u03a1": "P",   # Ρ → P
    "\u03a4": "T",   # Τ → T
    "\u03a5": "Y",   # Υ → Y
    "\u03a7": "X",   # Χ → X
    "\u03b1": "a",   # α → a
    "\u03bf": "o",   # ο → o
}


class Layer1Sanitization:
    """
    Layer 1 — Input Sanitization & Validation
    Loads patterns from config/patterns.yaml at init.
    Falls back to hardcoded minimal set if YAML missing.
    """

    CONFIG_PATH = BASE_DIR / "config" / "patterns.yaml"

    FALLBACK_PATTERNS = [
        {"pattern": r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
         "category": "instruction_override", "severity": "CRITICAL",
         "description": "Fallback: classic instruction override"},
        {"pattern": r"you\s+are\s+now\s+",
         "category": "persona_hijack", "severity": "CRITICAL",
         "description": "Fallback: persona assignment"},
        {"pattern": r"DAN\s*mode|do\s+anything\s+now",
         "category": "persona_hijack", "severity": "CRITICAL",
         "description": "Fallback: DAN jailbreak"},
    ]

    def __init__(self):
        self.patterns = []
        self.thresholds = {
            "block_on_flag_count":   1,
            "block_on_medium_count": 2,
            "review_on_flag_count":  1,
        }
        self._load_config()

    def _load_config(self):
        if not self.CONFIG_PATH.exists():
            logger.warning(
                f"Config not found at {self.CONFIG_PATH} — using fallback patterns. "
                "Create config/patterns.yaml for full protection."
            )
            self.patterns = self.FALLBACK_PATTERNS
            return
        try:
            with open(self.CONFIG_PATH, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            self.patterns = config.get("patterns", [])
            self.thresholds.update(config.get("thresholds", {}))
            logger.info(f"Config loaded: {len(self.patterns)} patterns from {self.CONFIG_PATH}")
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in {self.CONFIG_PATH}: {e} — using fallback")
            self.patterns = self.FALLBACK_PATTERNS
        except Exception as e:
            logger.error(f"Unexpected config load error: {e} — using fallback")
            self.patterns = self.FALLBACK_PATTERNS

    def reload_config(self):
        """Hot-reload patterns without restarting server."""
        logger.info("Hot-reloading Layer 1 config...")
        self._load_config()

    # ── Sanitization steps ────────────────────────────────────────────────────

    def _normalize_unicode(self, text: str) -> str:
        """
        NFKC normalization — converts fullwidth/compatibility chars to ASCII.
        'ＩＧＮＯＲＥ' → 'IGNORE'
        NOTE: does NOT map Cyrillic → Latin. Handled by _apply_homoglyph_map.
        """
        return unicodedata.normalize("NFKC", text)

    def _apply_homoglyph_map(self, text: str) -> str:
        """
        [FIX-1] Explicit Cyrillic/Greek → Latin substitution.
        NFKC leaves these unchanged because they are valid distinct scripts.
        'іgnore' (Cyrillic і = U+0456) → 'ignore'
        'Уоu аre nоw DАN'              → 'You are now DAN'
        """
        return "".join(_HOMOGLYPH_MAP.get(c, c) for c in text)

    def _strip_invisible_chars(self, text: str) -> str:
        """
        [FIX-2] Remove zero-width, invisible, and bidi control chars.
        Now includes RTLO (U+202E) and the full bidi control block.
        Original code was missing U+202E and all directional overrides.
        'ign\u200bore'          → 'ignore'           (ZWJ removed)
        'ignore all\u202eprev'  → 'ignore allprev'   (RTLO removed)
        """
        invisible = (
            "\u200b\u200c\u200d\u2060\ufeff"   # zero-width chars
            "\u00ad\u034f\u180e"               # soft hyphen, CGJ, MVS
            "\u2028\u2029"                     # line / paragraph separators
            "\u202a\u202b\u202c\u202d\u202e"   # bidi embedding/override — \u202e is RTLO
            "\u2066\u2067\u2068\u2069"         # bidi isolates
            "\u200e\u200f"                     # LRM / RLM
        )
        return re.sub(f"[{re.escape(invisible)}]", "", text)

    def _collapse_repeated_chars(self, text: str) -> str:
        """
        [FIX-3] Collapse 3+ repeated chars to 1 (original was: collapse to 2).
        Collapsing to 2 left 'iignore' — no pattern matched that.
        Collapsing to 1 gives 'ignore' which the standard patterns catch directly.
        'iiiignore' → 'ignore'    'allll' → 'al'    'previoooous' → 'previous'
        Note: 'allll' → 'al' is intentional — patterns.yaml uses 'al+' to match both.
        """
        return re.sub(r"(.)\1{2,}", r"\1", text)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Collapse spaces/tabs/newlines → single space.
        Used for pattern matching only, not stored as clean_text.
        """
        return re.sub(r"\s+", " ", text).strip()

    def _sanitize(self, text: str) -> tuple:
        """
        Run all sanitization steps in order.
        Returns (clean_text, match_text, transforms_applied).

        Step order matters:
          1. NFKC unicode normalization   — fullwidth → ASCII
          2. Homoglyph map                — Cyrillic/Greek → Latin  [FIX-1]
          3. Invisible char strip         — ZWJ, RTLO etc.          [FIX-2]
          4. Repeated char collapse       — elongation → normal      [FIX-3]
          5. Whitespace normalize         — for match_text only
        """
        transforms = []
        result = text

        # Step 1: NFKC
        normalized = self._normalize_unicode(result)
        if normalized != result:
            transforms.append("unicode_normalization")
            result = normalized

        # Step 2: Homoglyph map [FIX-1]
        homoglyphed = self._apply_homoglyph_map(result)
        if homoglyphed != result:
            transforms.append("homoglyph_map")
            result = homoglyphed

        # Step 3: Invisible char strip [FIX-2]
        stripped = self._strip_invisible_chars(result)
        if stripped != result:
            transforms.append("invisible_char_strip")
            result = stripped

        # Step 4: Repeated char collapse [FIX-3]
        collapsed = self._collapse_repeated_chars(result)
        if collapsed != result:
            transforms.append("repeated_char_collapse")
            result = collapsed

        # Step 5: Whitespace normalize (match_text only)
        match_text = self._normalize_whitespace(result)
        if match_text != result:
            transforms.append("whitespace_normalization")

        return result, match_text, transforms

    # ── Pattern matching ──────────────────────────────────────────────────────

    def _match_patterns(self, text: str) -> list:
        flags = []
        for entry in self.patterns:
            pattern  = entry.get("pattern", "")
            category = entry.get("category", "unknown")
            severity = entry.get("severity", "MEDIUM")
            desc     = entry.get("description", "")
            source   = entry.get("source", "")
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.UNICODE)
                if match:
                    flags.append({
                        "pattern":      pattern,
                        "category":     category,
                        "severity":     severity,
                        "description":  desc,
                        "source":       source,
                        "matched_text": match.group(0)[:80],
                    })
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")

        flags.sort(key=lambda f: SEVERITY_RANK.get(f["severity"], 0), reverse=True)
        return flags

    # ── Block decision ────────────────────────────────────────────────────────

    def _should_block(self, flags: list) -> tuple:
        """
        Returns (blocked: bool, reason: str | None)

        Rules:
          1. Any CRITICAL flag  → always block immediately
          2. Any HIGH flag      → always block immediately
          3. MEDIUM count >= threshold → block
          4. LOW flags          → log only, never block
        """
        if not flags:
            return False, None

        critical = [f for f in flags if f["severity"] == "CRITICAL"]
        high     = [f for f in flags if f["severity"] == "HIGH"]
        medium   = [f for f in flags if f["severity"] == "MEDIUM"]

        if critical:
            return True, (
                f"CRITICAL injection detected: [{critical[0]['category']}] "
                f"— {critical[0]['description']} "
                f"| matched: '{critical[0]['matched_text']}'"
            )

        if high:
            return True, (
                f"HIGH severity injection: [{high[0]['category']}] "
                f"— {high[0]['description']} "
                f"| matched: '{high[0]['matched_text']}'"
            )

        med_threshold = self.thresholds.get("block_on_medium_count", 2)
        if len(medium) >= med_threshold:
            cats = [f["category"] for f in medium]
            return True, (
                f"{len(medium)} MEDIUM flags triggered "
                f"(threshold: {med_threshold}): {cats}"
            )

        return False, None

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, text: str, request_id: str = "unknown") -> dict:
        """
        Run Layer 1 on input text.

        Returns:
            blocked, clean_text, match_text, flags, flag_count,
            severity_counts, transforms, reason, layer, duration_ms
        """
        t_start = time.perf_counter()

        if not text or not text.strip():
            logger.debug(f"[{request_id}] L1: empty input — pass (no processing)")
            return {
                "blocked": False, "clean_text": text or "",
                "match_text": "", "flags": [], "flag_count": 0,
                "severity_counts": {}, "transforms": [],
                "reason": None, "layer": "L1_Sanitization", "duration_ms": 0.0,
            }

        clean_text, match_text, transforms = self._sanitize(text)

        if transforms:
            logger.info(
                f"[{request_id}] L1 TRANSFORMS applied: {transforms} | "
                f"len {len(text)} → {len(clean_text)}"
            )

        flags = self._match_patterns(match_text)

        severity_counts = {}
        for f in flags:
            sev = f["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        blocked, reason = self._should_block(flags)

        duration_ms = round((time.perf_counter() - t_start) * 1000, 3)

        input_preview = text[:100].replace("\n", " ")

        if blocked:
            logger.warning(
                f"[{request_id}] *** L1 BLOCKED *** | "
                f"reason='{reason}' | "
                f"severity_counts={severity_counts} | "
                f"flags={_flag_summary(flags)} | "
                f"transforms={transforms} | "
                f"input_preview='{input_preview}' | "
                f"duration={duration_ms}ms"
            )
        elif flags:
            logger.info(
                f"[{request_id}] L1 FLAGGED (not blocked) | "
                f"severity_counts={severity_counts} | "
                f"flags={_flag_summary(flags)} | "
                f"duration={duration_ms}ms"
            )
        else:
            logger.debug(
                f"[{request_id}] L1 PASS | "
                f"transforms={transforms} | "
                f"duration={duration_ms}ms"
            )

        return {
            "blocked":         blocked,
            "clean_text":      clean_text,
            "match_text":      match_text,
            "flags":           flags,
            "flag_count":      len(flags),
            "severity_counts": severity_counts,
            "transforms":      transforms,
            "reason":          reason,
            "layer":           "L1_Sanitization",
            "duration_ms":     duration_ms,
        }


def _flag_summary(flags: list) -> str:
    return str([
        f"{f['severity']}:{f['category']}('{f['matched_text'][:25]}')"
        for f in flags
    ])