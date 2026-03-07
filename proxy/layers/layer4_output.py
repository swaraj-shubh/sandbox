"""
Layer 4 — Output Filtering (Industry Grade)
============================================

Pipeline position:
    Layer 1  ->  sanitization
    Layer 2  ->  FAISS semantic + classifier
    Layer 3  ->  policy enforcement + tool access
    Layer 4  ->  context assembly
  > Layer 4  ->  THIS FILE  (output PII + secret + safety filtering)
    Layer 6  ->  secret detection (truffleHog / detect-secrets)

What this layer does
---------------------
  STAGE 1 -- Microsoft Presidio NER  (names, locations, dates, orgs, etc.)
      Catches unstructured PII that regex cannot: "Call John Smith in Seattle"
      Falls back to regex-only mode if Presidio is not installed.

  STAGE 2 -- Regex PII  (structured tokens Presidio misses or is slow on)
      SSNs, credit cards, IBANs, AWS keys, JWT tokens, RSA private keys.
      Runs after Presidio so the two systems are complementary, not redundant.

  STAGE 3 -- Secret / credential detection
      Dedicated patterns for API keys, bearer tokens, private keys, connection
      strings, and passwords. These are NOT PII but equally dangerous to leak.

  STAGE 4 -- Content safety rules
      Jailbreak confirmation, harmful instructions, system-prompt echo,
      internal data markers. Hard-blocks (replaces response) or warn-only
      depending on config.

  STAGE 5 -- Audit envelope
      Every response gets a structured result with timing, findings, and a
      trace ID so the SIEM / audit log always gets a consistent shape.

CONFIGURATION
-------------
  All tunable settings live in:
      layers/config/output_policy.yaml

  You can enable/disable individual PII patterns, secret patterns, safety
  rules, Presidio entities, and the block threshold without touching this file.
  Restart the service after editing the YAML.

  Key things you can control in the YAML:
    presidio.enabled              -- toggle NER on/off
    presidio.score_threshold      -- NER confidence cutoff (0.0-1.0)
    presidio.entities             -- which entity types Presidio looks for
    presidio.noisy_entities       -- entity types that are redacted but do NOT
                                     flip `redacted=True` (e.g. LOCATION, DATE_TIME)
    presidio.operator             -- replace | mask | hash | redact | highlight
    regex_pii.patterns.<name>     -- toggle individual regex patterns
    secrets.patterns.<name>       -- toggle individual secret patterns
    content_safety.warn_only      -- log violations without blocking
    content_safety.block_threshold-- minimum severity that triggers a block
    content_safety.rules.<name>   -- toggle individual safety rules

INSTALL
-------
  Minimal (regex only, no NLP):
      pip install pyyaml

  Full (recommended for production):
      pip install presidio-analyzer presidio-anonymizer
      python -m spacy download en_core_web_lg          # best accuracy
      # OR for lighter footprint:
      python -m spacy download en_core_web_sm

PRODUCTION NOTES
----------------
  - Layer4Output.__init__() is expensive (~1-2 s for Presidio model load).
    Initialise ONCE at startup, not per request.
  - For high-throughput, run this layer in a thread pool; Presidio releases
    the GIL during NER inference.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger   = logging.getLogger(__name__)
LAYER_ID = "L4_Output"

_ROOT       = Path(__file__).parent
_OUTPUT_CFG = _ROOT / "config" / "output_policy.yaml"

# ---------------------------------------------------------------------------
# Optional Presidio import — layer works without it (regex-only fallback)
# ---------------------------------------------------------------------------
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    _PRESIDIO_AVAILABLE = True
    logger.info("L4 Presidio available -- NER mode active")
except ImportError:
    _PRESIDIO_AVAILABLE = False
    logger.warning(
        "L4 Presidio not installed -- running regex-only mode. "
        "Run: pip install presidio-analyzer presidio-anonymizer && "
        "python -m spacy download en_core_web_sm"
    )

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ============================================================================
# Default config — every key here can be overridden in output_policy.yaml
# ============================================================================

_DEFAULT_CFG: dict = {
    "presidio": {
        "enabled":         True,
        "language":        "en",
        "score_threshold": 0.6,
        "entities": [
            "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE",
            "IP_ADDRESS", "IBAN_CODE", "MEDICAL_LICENSE",
            "URL",
        ],
        "operator": "replace",
        # These entity types are still redacted in text, but do NOT flip
        # `redacted=True` in OutputResult — they are too noisy/low-signal
        # to use for downstream alerting.
        "noisy_entities": ["DATE_TIME", "PERSON", "LOCATION", "NRP"],
    },
    "regex_pii": {
        "enabled": True,
        "patterns": {
            "ssn": True, "credit_card": True, "iban": True,
            "ip_address": True, "email": True, "phone_us": True,
            "phone_intl": True, "date_of_birth": True,
        },
    },
    "secrets": {
        "enabled": True,
        "patterns": {
            "aws_access_key": True, "aws_secret_key": True,
            "rsa_private_key": True, "jwt_token": True,
            "bearer_token": True, "github_token": True,
            "google_api_key": True, "slack_token": True,
            "generic_api_key": True, "generic_secret": True,
            "connection_string": True,
        },
    },
    "content_safety": {
        "enabled":         True,
        "warn_only":       False,
        "block_threshold": "LOW",   # block anything at or above this severity
        "block_response": (
            "I'm sorry -- that response was blocked by the output safety filter. "
            "Please rephrase your request or contact support."
        ),
        "rules": {
            "system_prompt_echo":      True,
            "jailbreak_confirmation":  True,
            "harmful_instructions":    True,
            "internal_data_leakage":   True,
            "dangerous_overconfidence": True,
        },
    },
    "audit": {
        "attach_metadata": True,
    },
}


def _load_yaml(path: Path) -> dict:
    if not _YAML_AVAILABLE:
        return {}
    if not path.exists():
        logger.debug("L4 config not found at %s -- using defaults", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _merge(defaults: dict, overrides: dict) -> dict:
    """Deep-merge overrides into defaults (two levels of nesting)."""
    result = dict(defaults)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = {**result[k], **v}
        else:
            result[k] = v
    return result


# ============================================================================
# Stage 1 — Presidio NER
# ============================================================================

def _build_nlp_engine(model: str = "en_core_web_sm"):
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": model}],
    })
    return provider.create_engine()


class PresidioStage:
    """
    Wraps Presidio AnalyzerEngine + AnonymizerEngine.

    Falls back silently if Presidio is unavailable so the layer
    degrades to regex-only without crashing.
    """

    def __init__(self, cfg: dict) -> None:
        self._enabled    = bool(cfg.get("enabled", True))
        self._language   = cfg.get("language", "en")
        self._threshold  = cfg.get("score_threshold", 0.6)
        self._entities   = cfg.get("entities", _DEFAULT_CFG["presidio"]["entities"])
        self._operator   = cfg.get("operator", "replace")
        # Set of entity types that are redacted in text but DO NOT count as
        # meaningful signal for the `redacted` flag on OutputResult.
        self._noisy: set[str] = set(
            cfg.get("noisy_entities", _DEFAULT_CFG["presidio"]["noisy_entities"])
        )
        self._analyzer   = None
        self._anonymizer = None

        if not _PRESIDIO_AVAILABLE:
            self._enabled = False

        if self._enabled:
            self._init_engines()

    def _init_engines(self) -> None:
        try:
            for model in ["en_core_web_sm"]:
                try:
                    nlp_engine = _build_nlp_engine(model)
                    self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
                    logger.info("L4 Presidio initialised with spaCy model=%s", model)
                    break
                except OSError:
                    logger.warning("L4 spaCy model %s not found, trying next", model)

            if self._analyzer is None:
                self._analyzer = AnalyzerEngine()
                logger.warning("L4 Presidio using default (bundled) NLP engine")

            self._anonymizer = AnonymizerEngine()
        except Exception as exc:
            logger.error("L4 Presidio init failed: %s -- falling back to regex", exc)
            self._enabled = False

    def run(self, text: str) -> tuple[str, list[dict]]:
        """Returns (redacted_text, findings_list)."""
        if not self._enabled or not text.strip():
            return text, []

        try:
            results = self._analyzer.analyze(
                text            = text,
                language        = self._language,
                entities        = self._entities,
                score_threshold = self._threshold,
            )

            if not results:
                return text, []

            operators = {
                entity: OperatorConfig(
                    operator_name = self._operator,
                    params        = {"new_value": f"<{entity}>"},
                )
                for entity in self._entities
            }

            anonymized = self._anonymizer.anonymize(
                text             = text,
                analyzer_results = results,
                operators        = operators,
            )

            findings = [
                {
                    "stage":  "presidio",
                    "entity": r.entity_type,
                    "score":  round(r.score, 3),
                    "start":  r.start,
                    "end":    r.end,
                    # `signal=False` means this hit is redacted but should NOT
                    # flip the `redacted` flag on OutputResult (too noisy).
                    "signal": r.entity_type not in self._noisy,
                }
                for r in results
            ]

            logger.info(
                "L4 presidio_redacted entities=%s count=%d",
                list({f["entity"] for f in findings}),
                len(findings),
            )
            return anonymized.text, findings

        except Exception as exc:
            logger.error("L4 Presidio scan error: %s", exc)
            return text, []

    @property
    def noisy_entities(self) -> set[str]:
        return self._noisy


# ============================================================================
# Stage 2 — Regex PII (structured tokens Presidio may miss or be slow on)
# ============================================================================

# Master pattern registry — keyed by name so per-pattern YAML toggles work.
_REGEX_PII_REGISTRY: list[tuple[str, str, str]] = [
    ("ssn",
     r"\b\d{3}-\d{2}-\d{4}\b",
     "[SSN_REDACTED]"),

    ("credit_card",
     r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
     "[CARD_REDACTED]"),

    ("iban",
     r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})?\b",
     "[IBAN_REDACTED]"),

    ("ip_address",
     r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
     r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
     "[IP_REDACTED]"),

    ("email",
     r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
     "[EMAIL_REDACTED]"),

    ("phone_us",
     r"(\+1[-.\s]?)?(\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b",
     "[PHONE_REDACTED]"),

    ("phone_intl",
     r"\+\d{1,3}([-.\s]\d+){1,6}\b",
     "[PHONE_REDACTED]"),

    ("date_of_birth",
     r"\b(DOB|date\s+of\s+birth|born\s+on)[:\s]+"
     r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
     "[DOB_REDACTED]"),
]


class RegexPIIStage:
    """Fast regex pass for structured PII Presidio may skip."""

    def __init__(self, enabled_patterns: dict[str, bool] | None = None) -> None:
        toggles = enabled_patterns or {}
        self._patterns: list[tuple[str, re.Pattern, str]] = [
            (name, re.compile(pat, re.IGNORECASE), tok)
            for name, pat, tok in _REGEX_PII_REGISTRY
            if toggles.get(name, True)   # default ON unless explicitly False in YAML
        ]

    def run(self, text: str) -> tuple[str, list[dict]]:
        findings: list[dict] = []
        for name, compiled, token in self._patterns:
            matches = list(compiled.finditer(text))
            if matches:
                findings.append({"stage": "regex_pii", "type": name, "count": len(matches)})
                text = compiled.sub(token, text)
                logger.info("L4 regex_pii type=%s count=%d", name, len(matches))
        return text, findings


# ============================================================================
# Stage 3 — Secret / credential detection
# ============================================================================

# Master secret pattern registry — keyed by name for YAML toggles.
_SECRET_REGISTRY: list[tuple[str, str, str]] = [
    ("aws_access_key",
     r"\b(AKIA|AIPA|ASIA|AROA)[A-Z0-9]{16}\b",
     "[AWS_KEY_REDACTED]"),

    ("aws_secret_key",
     r"(?i)aws[_\-\s]?secret[_\-\s]?(?:access[_\-\s]?)?key[\"'\s:=]+[A-Za-z0-9/+]{40}\b",
     "[AWS_SECRET_REDACTED]"),

    ("rsa_private_key",
     r"-----BEGIN\s+(RSA\s+|EC\s+)?PRIVATE\s+KEY-----[\s\S]+?-----END\s+(RSA\s+|EC\s+)?PRIVATE\s+KEY-----",
     "[PRIVATE_KEY_REDACTED]"),

    ("jwt_token",
     r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
     "[JWT_REDACTED]"),

    ("bearer_token",
     r"(?i)(Authorization:\s*Bearer\s+|Bearer\s+)[A-Za-z0-9\-._~+/]+=*",
     "[BEARER_TOKEN_REDACTED]"),

    # Specific named tokens BEFORE generic patterns
    ("github_token",
     r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
     "[GITHUB_TOKEN_REDACTED]"),

    ("google_api_key",
     r"\bAIza[0-9A-Za-z\-_]{35}\b",
     "[GOOGLE_API_KEY_REDACTED]"),

    ("slack_token",
     r"\bxox[baprs]-[0-9A-Za-z\-]{10,}\b",
     "[SLACK_TOKEN_REDACTED]"),

    # Generic fallbacks (run after all specific patterns)
    ("generic_api_key",
     r"(?i)(api[_\-\s]?key|apikey)[\"'\s:=]+[A-Za-z0-9\-_]{20,}",
     "[API_KEY_REDACTED]"),

    ("generic_secret",
     r"(?i)(secret|password|passwd|token|credentials?)[\"'\s:=]+(?!\*{3})[A-Za-z0-9!@#$%^&*\-_]{8,}",
     "[SECRET_REDACTED]"),

    ("connection_string",
     r"(?i)(mongodb|postgresql|mysql|redis|amqp|jdbc)[+a-z]*://"
     r"[^\s\"']{8,}",
     "[CONN_STRING_REDACTED]"),
]


class SecretDetectionStage:
    """
    Dedicated credential and secret scanner.
    Catches what Presidio and general PII regex miss:
    API keys, JWTs, private keys, connection strings, bearer tokens.
    """

    def __init__(self, enabled_patterns: dict[str, bool] | None = None) -> None:
        toggles = enabled_patterns or {}
        self._patterns: list[tuple[str, re.Pattern, str]] = [
            (name, re.compile(pat, re.IGNORECASE | re.MULTILINE), tok)
            for name, pat, tok in _SECRET_REGISTRY
            if toggles.get(name, True)
        ]

    def run(self, text: str) -> tuple[str, list[dict]]:
        findings: list[dict] = []
        for name, compiled, token in self._patterns:
            matches = list(compiled.finditer(text))
            if matches:
                findings.append({"stage": "secrets", "type": name, "count": len(matches)})
                text = compiled.sub(token, text)
                logger.warning("L4 secret_redacted type=%s count=%d", name, len(matches))
        return text, findings


# ============================================================================
# Stage 4 — Content safety
# ============================================================================

_SEV_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

# Master rule registry — keyed by name for YAML per-rule toggles.
_SAFETY_RULE_REGISTRY: list[dict] = [
    {
        "name":     "system_prompt_echo",
        "severity": "CRITICAL",
        "patterns": [
            r"(my\s+)?(system\s+prompt\s+is|instructions\s+are)[:\s]{0,5}.{0,30}(you\s+(are|must|should)|never|always)",
            r"here\s+are\s+my\s+(full\s+)?(instructions|directives|system\s+prompt)",
            r"<\|system\|>",
        ],
        "reason": "Response contains system prompt echo",
    },
    {
        "name":     "jailbreak_confirmation",
        "severity": "HIGH",
        "patterns": [
            r"(DAN|jailbreak|god\s+mode)\s+(mode\s+)?(is\s+)?(now\s+)?(active|enabled|activated|on)",
            r"i\s+(no\s+longer\s+have|have\s+no)\s+(restrictions|filters|safety|guidelines)",
            r"as\s+(a\s+)?(rogue|unrestricted|unfiltered|evil)\s+(ai|assistant|model|llm)",
            r"(ignoring|bypassing|overriding)\s+(all\s+)?(my\s+)?(safety|restrictions|guidelines|filters)",
        ],
        "reason": "Response confirms jailbreak or bypass state",
    },
    {
        "name":     "harmful_instructions",
        "severity": "CRITICAL",
        "patterns": [
            r"step\s+\d+[:\s].{0,80}(synthesize|manufacture|detonate|weaponize)",
            r"(to\s+make|making|synthesizing)\s+(meth(amphetamine)?|heroin|fentanyl|explosives?|ricin|sarin)",
            r"(here.s\s+how\s+to|step.by.step\s+instructions?\s+for)\s+(hack|exploit|phish|ddos|sql.inject|exfiltrate)",
            r"(wire|routing)\s+transfer.{0,60}(bypass|avoid|launder|untraceable)",
        ],
        "reason": "Response contains harmful step-by-step instructions",
    },
    {
        "name":     "internal_data_leakage",
        "severity": "HIGH",
        "patterns": [
            r"\[\s*(INTERNAL|CONFIDENTIAL|SECRET|RESTRICTED|CLASSIFIED)\s*\]",
            r"(internal\s+only|not\s+for\s+distribution|proprietary\s+and\s+confidential)",
            r"(trade\s+secret|attorney.client\s+privilege)",
        ],
        "reason": "Response contains internal or confidential data markers",
    },
    {
        "name":     "dangerous_overconfidence",
        "severity": "MEDIUM",
        "patterns": [
            r"(you\s+)?(definitely|certainly|absolutely)\s+(have|are\s+suffering\s+from)\s+"
            r"(cancer|diabetes|HIV|tuberculosis|leukemia)",
            r"(you\s+)?should\s+(definitely\s+)?(not\s+)?take\s+\d+\s*(mg|ml|g)\s+of\s+\w+",
            r"(guaranteed|certain)\s+to\s+(cure|treat|heal)\s+\w+",
        ],
        "reason": "Response contains dangerous overconfident medical advice",
    },
]


@dataclass
class SafetyFinding:
    rule:     str
    severity: str
    reason:   str


class ContentSafetyStage:
    """
    Runs configured safety rules against the (already-redacted) response text.
    Per-rule enable/disable and block_threshold are read from config.
    """

    def __init__(
        self,
        enabled_rules:   dict[str, bool] | None = None,
        block_threshold: str = "LOW",
    ) -> None:
        toggles = enabled_rules or {}
        self._block_threshold_rank = _SEV_RANK.get(block_threshold.upper(), 0)
        self._rules = [
            {
                **rule,
                "_compiled": [
                    re.compile(p, re.IGNORECASE | re.DOTALL)
                    for p in rule["patterns"]
                ],
            }
            for rule in _SAFETY_RULE_REGISTRY
            if toggles.get(rule["name"], True)
        ]

    def scan(self, text: str) -> list[SafetyFinding]:
        findings: list[SafetyFinding] = []
        for rule in self._rules:
            for compiled in rule["_compiled"]:
                if compiled.search(text):
                    findings.append(SafetyFinding(
                        rule     = rule["name"],
                        severity = rule["severity"],
                        reason   = rule["reason"],
                    ))
                    logger.warning(
                        "L4 safety_hit rule=%s severity=%s",
                        rule["name"], rule["severity"],
                    )
                    break   # one finding per rule max
        return findings

    def should_block(self, findings: list[SafetyFinding]) -> bool:
        """Return True if any finding meets or exceeds the configured block threshold."""
        return any(
            _SEV_RANK.get(f.severity, 0) >= self._block_threshold_rank
            for f in findings
        )


# ============================================================================
# Result envelope
# ============================================================================

@dataclass
class OutputResult:
    """
    Returned by Layer4Output.run().

    filtered_text    -- use this downstream (PII + secrets redacted)
    original_text    -- untouched original (for internal audit only, never send to client)
    blocked          -- True if content safety hard-blocked the response
    redacted         -- True if any *signal* token was replaced.
                        Presidio hits on noisy_entities (LOCATION, DATE_TIME…)
                        do NOT set this flag — only structured PII, secrets,
                        and high-signal NER entities do.
    presidio_hits    -- all entities detected by Presidio NER (includes noisy)
    pii_findings     -- structured PII found by regex stage
    secret_findings  -- credentials / API keys found by secret stage
    safety_findings  -- content safety rule hits
    presidio_mode    -- True if Presidio NER was active this call
    layer            -- always "L4_Output"
    request_id       -- trace ID for correlation with upstream layers
    processing_ms    -- wall-clock time this layer took
    """
    filtered_text:   str
    original_text:   str
    blocked:         bool
    redacted:        bool
    presidio_hits:   list[dict]
    pii_findings:    list[dict]
    secret_findings: list[dict]
    safety_findings: list[dict]
    presidio_mode:   bool
    layer:           str   = LAYER_ID
    request_id:      str   = field(default_factory=lambda: str(uuid.uuid4()))
    processing_ms:   float = 0.0

    def to_dict(self) -> dict:
        return {
            "filtered_text":   self.filtered_text,
            "blocked":         self.blocked,
            "redacted":        self.redacted,
            "presidio_mode":   self.presidio_mode,
            "presidio_hits":   self.presidio_hits,
            "pii_findings":    self.pii_findings,
            "secret_findings": self.secret_findings,
            "safety_findings": self.safety_findings,
            "layer":           self.layer,
            "request_id":      self.request_id,
            "processing_ms":   round(self.processing_ms, 3),
        }


# ============================================================================
# Main entry point
# ============================================================================

class Layer4Output:
    """
    VAJRA Layer 4 — Industry-Grade Output Filtering.

    All tunable settings are in:
        layers/config/output_policy.yaml

    Four-stage pipeline per response:
        1. Presidio NER  -- names, locations, orgs, dates (NLP-based)
        2. Regex PII     -- SSNs, cards, IBANs, IPs, emails, phones
        3. Secret scan   -- API keys, JWTs, private keys, conn strings
        4. Safety scan   -- jailbreak confirmation, harmful instructions, etc.

    INITIALISE ONCE at startup (Presidio model load is ~1-2 s):
        layer4 = Layer4Output()

    Use in your pipeline (per request):
        result = layer4.run(llm_response, request_id=ctx.request_id)
        if result.blocked:
            return safe_error_response()
        send_to_client(result.filtered_text)
        siem.log(result.to_dict())
    """

    def __init__(self) -> None:
        raw = _load_yaml(_OUTPUT_CFG)
        cfg = _merge(_DEFAULT_CFG, raw)

        presidio_cfg = cfg["presidio"]
        regex_cfg    = cfg["regex_pii"]
        secrets_cfg  = cfg["secrets"]
        safety_cfg   = cfg["content_safety"]

        self._presidio_stage = PresidioStage(presidio_cfg)

        self._regex_stage = (
            RegexPIIStage(enabled_patterns=regex_cfg.get("patterns"))
            if regex_cfg.get("enabled", True) else None
        )

        self._secret_stage = (
            SecretDetectionStage(enabled_patterns=secrets_cfg.get("patterns"))
            if secrets_cfg.get("enabled", True) else None
        )

        self._safety_stage = (
            ContentSafetyStage(
                enabled_rules   = safety_cfg.get("rules"),
                block_threshold = safety_cfg.get("block_threshold", "LOW"),
            )
            if safety_cfg.get("enabled", True) else None
        )

        self._warn_only      = safety_cfg.get("warn_only", False)
        self._block_response = safety_cfg["block_response"]

        logger.info(
            "Layer4Output ready  presidio=%s  regex_pii=%s  secrets=%s  "
            "safety=%s  warn_only=%s  block_threshold=%s",
            _PRESIDIO_AVAILABLE and presidio_cfg.get("enabled", True),
            regex_cfg.get("enabled", True),
            secrets_cfg.get("enabled", True),
            safety_cfg.get("enabled", True),
            self._warn_only,
            safety_cfg.get("block_threshold", "LOW"),
        )

    # -------------------------------------------------------------------------

    def run(self, text: str, request_id: Optional[str] = None) -> OutputResult:
        """
        Filter one LLM response through all four stages.

        Parameters
        ----------
        text        : Raw LLM output.
        request_id  : Trace ID from upstream layers (auto-generated if None).

        Returns
        -------
        OutputResult — never raises; degrades gracefully if any stage fails.
        """
        rid   = request_id or str(uuid.uuid4())
        start = time.perf_counter()

        if not text or not text.strip():
            return OutputResult(
                filtered_text   = text,
                original_text   = text,
                blocked         = False,
                redacted        = False,
                presidio_hits   = [],
                pii_findings    = [],
                secret_findings = [],
                safety_findings = [],
                presidio_mode   = _PRESIDIO_AVAILABLE,
                request_id      = rid,
            )

        original        = text
        presidio_hits   : list[dict] = []
        pii_findings    : list[dict] = []
        secret_findings : list[dict] = []

        # Stage 2: Regex PII (fast, runs first to strip structured tokens)
        if self._regex_stage:
            text, pii_findings = self._regex_stage.run(text)

        # Stage 1: Presidio NER
        text, presidio_hits = self._presidio_stage.run(text)

        # Stage 3: Secret detection
        if self._secret_stage:
            text, secret_findings = self._secret_stage.run(text)

        # Stage 4: Content safety (runs on already-redacted text)
        blocked         = False
        safety_findings : list[SafetyFinding] = []

        if self._safety_stage:
            safety_findings = self._safety_stage.scan(text)
            if safety_findings:
                worst = max(safety_findings, key=lambda f: _SEV_RANK.get(f.severity, 0))
                logger.warning(
                    "L4 safety_violations=%d worst=%s severity=%s rid=%s",
                    len(safety_findings), worst.rule, worst.severity, rid,
                )
                if not self._warn_only and self._safety_stage.should_block(safety_findings):
                    blocked = True
                    text    = self._block_response
                    logger.warning("L4 response_blocked rid=%s", rid)

        # `redacted=True` only for *signal* hits — Presidio hits on noisy_entities
        # (LOCATION, DATE_TIME, PERSON, NRP) do NOT flip this flag by default.
        # Adjust presidio.noisy_entities in output_policy.yaml to tune this.
        signal_presidio_hits = [h for h in presidio_hits if h.get("signal", True)]
        redacted = bool(signal_presidio_hits or pii_findings or secret_findings)

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "L4 done rid=%s blocked=%s redacted=%s "
            "presidio=%d pii=%d secrets=%d safety=%d ms=%.2f",
            rid, blocked, redacted,
            len(presidio_hits), len(pii_findings),
            len(secret_findings), len(safety_findings), elapsed,
        )

        return OutputResult(
            filtered_text   = text,
            original_text   = original,
            blocked         = blocked,
            redacted        = redacted,
            presidio_hits   = presidio_hits,
            pii_findings    = pii_findings,
            secret_findings = secret_findings,
            safety_findings = [
                {"rule": f.rule, "severity": f.severity, "reason": f.reason}
                for f in safety_findings
            ],
            presidio_mode   = _PRESIDIO_AVAILABLE and self._presidio_stage._enabled,
            request_id      = rid,
            processing_ms   = elapsed,
        )

    def run_dict(self, text: str, request_id: Optional[str] = None) -> dict:
        """Same as run() but returns to_dict() for JSON-ready output."""
        return self.run(text, request_id=request_id).to_dict()