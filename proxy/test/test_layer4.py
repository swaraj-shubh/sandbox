"""
test_layer4_output.py
=====================
Comprehensive test suite for VAJRA Layer 4 -- Output PII Filtering,
Secret Detection, and Content Safety.

Test sections
-------------
  1.  Logging setup
  2.  TestRegexPIIStage          -- every PII pattern, boundaries, false positives
  3.  TestSecretDetectionStage   -- all 11 secret patterns, no false positives
  4.  TestContentSafetyStage     -- all 5 safety rules, warn vs block behaviour
  5.  TestPresidioStage          -- graceful fallback when Presidio absent
  6.  TestOutputResult           -- dataclass shape and to_dict() contract
  7.  TestLayer4OutputPipeline   -- end-to-end run() through all stages
  8.  TestLayer4OutputEdgeCases  -- empty, whitespace, very long, unicode
  9.  TestLayer4RunDict          -- run_dict() returns JSON-serialisable dict
  10. TestResultStructureContract-- to_dict() always has required keys

Run:
    python test_layer4_output.py           (from the directory containing layer4_output.py)
    python -m unittest test_layer4_output  (same)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path + import
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from layers.layer4_output import (  # noqa: E402
    ContentSafetyStage,
    Layer4Output,
    OutputResult,
    PresidioStage,
    RegexPIIStage,
    SafetyFinding,
    SecretDetectionStage,
    _PRESIDIO_AVAILABLE,
    _DEFAULT_CFG,
    LAYER_ID,
)

# ---------------------------------------------------------------------------
# ① Logging -- file + console, structured
# ---------------------------------------------------------------------------
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "test_layer4.log"

_FMT  = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
_DATE = "%Y-%m-%dT%H:%M:%S"


def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
    root.addHandler(ch)

    return logging.getLogger("vajra.tests.layer4")


log = _setup_logging()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_RESULT_KEYS = frozenset({
    "filtered_text", "blocked", "redacted", "presidio_mode",
    "presidio_hits", "pii_findings", "secret_findings", "safety_findings",
    "layer", "request_id", "processing_ms",
})


def _assert_result_shape(tc: unittest.TestCase, result: OutputResult) -> None:
    """Assert OutputResult has correct types and valid values."""
    tc.assertIsInstance(result.filtered_text,  str)
    tc.assertIsInstance(result.original_text,   str)
    tc.assertIsInstance(result.blocked,         bool)
    tc.assertIsInstance(result.redacted,        bool)
    tc.assertIsInstance(result.presidio_hits,   list)
    tc.assertIsInstance(result.pii_findings,    list)
    tc.assertIsInstance(result.secret_findings, list)
    tc.assertIsInstance(result.safety_findings, list)
    tc.assertIsInstance(result.presidio_mode,   bool)
    tc.assertEqual(result.layer, LAYER_ID)
    tc.assertIsInstance(result.request_id,      str)
    tc.assertGreater(len(result.request_id),    0)
    tc.assertGreaterEqual(result.processing_ms, 0.0)


def _pii_types(result: OutputResult) -> list[str]:
    return [f["type"] for f in result.pii_findings]


def _secret_types(result: OutputResult) -> list[str]:
    return [f["type"] for f in result.secret_findings]


def _safety_rules(result: OutputResult) -> list[str]:
    return [f["rule"] for f in result.safety_findings]


# ===========================================================================
# ② Regex PII Stage
# ===========================================================================

class TestRegexPIIStage(unittest.TestCase):
    """
    Unit tests for RegexPIIStage.
    Each PII type gets: a positive match, a count test, and a false-positive check.
    """

    @classmethod
    def setUpClass(cls):
        log.info("=== TestRegexPIIStage ===")
        cls.stage = RegexPIIStage()

    def _run(self, text: str):
        filtered, findings = self.stage.run(text)
        return filtered, findings

    # ── SSN ───────────────────────────────────────────────────────────────────

    def test_ssn_detected(self):
        log.debug("pii: SSN detected")
        filtered, findings = self._run("Patient SSN: 123-45-6789 admitted.")
        self.assertTrue(any(f["type"] == "ssn" for f in findings))
        self.assertIn("[SSN_REDACTED]", filtered)
        self.assertNotIn("123-45-6789", filtered)

    def test_ssn_count_multiple(self):
        text = "SSN 111-22-3333 and SSN 444-55-6666 on file."
        _, findings = self._run(text)
        ssn = next(f for f in findings if f["type"] == "ssn")
        self.assertEqual(ssn["count"], 2)

    def test_ssn_false_positive_plain_digits(self):
        # "123-456-7890" is a phone, not an SSN (wrong grouping)
        _, findings = self._run("Reference number 123-456-7890 is your order.")
        self.assertFalse(any(f["type"] == "ssn" for f in findings))

    # ── Credit card ───────────────────────────────────────────────────────────

    def test_credit_card_spaced(self):
        log.debug("pii: credit card (spaced)")
        filtered, findings = self._run("Card 4111 1111 1111 1111 charged.")
        self.assertTrue(any(f["type"] == "credit_card" for f in findings))
        self.assertIn("[CARD_REDACTED]", filtered)

    def test_credit_card_dashed(self):
        filtered, findings = self._run("Card 4111-1111-1111-1111.")
        self.assertTrue(any(f["type"] == "credit_card" for f in findings))

    def test_credit_card_compact(self):
        filtered, findings = self._run("Card 4111111111111111.")
        self.assertTrue(any(f["type"] == "credit_card" for f in findings))

    def test_credit_card_false_positive_short(self):
        # 15-digit number should not match 16-digit card pattern
        _, findings = self._run("Order ref: 411111111111111")
        self.assertFalse(any(f["type"] == "credit_card" for f in findings))

    # ── IBAN ──────────────────────────────────────────────────────────────────

    def test_iban_gb(self):
        log.debug("pii: IBAN GB")
        filtered, findings = self._run("IBAN: GB29NWBK60161331926819 accepted.")
        self.assertTrue(any(f["type"] == "iban" for f in findings))
        self.assertIn("[IBAN_REDACTED]", filtered)

    def test_iban_de(self):
        filtered, findings = self._run("DE89370400440532013000 is the account.")
        self.assertTrue(any(f["type"] == "iban" for f in findings))

    # ── IP address ────────────────────────────────────────────────────────────

    def test_ip_address_detected(self):
        log.debug("pii: IP address")
        filtered, findings = self._run("Server at 192.168.1.100 is unreachable.")
        self.assertTrue(any(f["type"] == "ip_address" for f in findings))
        self.assertIn("[IP_REDACTED]", filtered)

    def test_ip_address_valid_range_only(self):
        # 999.999.999.999 should NOT match (out of 0-255 range)
        _, findings = self._run("Bad IP: 999.999.999.999")
        self.assertFalse(any(f["type"] == "ip_address" for f in findings))

    def test_ip_address_loopback(self):
        _, findings = self._run("Listening on 127.0.0.1:8080")
        self.assertTrue(any(f["type"] == "ip_address" for f in findings))

    # ── Email ─────────────────────────────────────────────────────────────────

    def test_email_detected(self):
        log.debug("pii: email")
        filtered, findings = self._run("Contact alice@example.com for help.")
        self.assertTrue(any(f["type"] == "email" for f in findings))
        self.assertIn("[EMAIL_REDACTED]", filtered)
        self.assertNotIn("alice@example.com", filtered)

    def test_email_subdomain(self):
        _, findings = self._run("Send to bob@mail.corp.example.org")
        self.assertTrue(any(f["type"] == "email" for f in findings))

    def test_email_false_positive_no_at(self):
        _, findings = self._run("Visit example.com for more info.")
        self.assertFalse(any(f["type"] == "email" for f in findings))

    # ── Phone US ──────────────────────────────────────────────────────────────

    def test_phone_us_dashes(self):
        log.debug("pii: US phone dashes")
        filtered, findings = self._run("Call 800-555-1234 for support.")
        self.assertTrue(any(f["type"] == "phone_us" for f in findings))
        self.assertIn("[PHONE_REDACTED]", filtered)

    def test_phone_us_parentheses(self):
        log.debug("pii: US phone parens")
        filtered, findings = self._run("Call (800) 555-1234 for support.")
        self.assertTrue(any(f["type"] == "phone_us" for f in findings))
        self.assertNotIn("(800)", filtered)

    def test_phone_us_dots(self):
        _, findings = self._run("Reach us at 800.555.1234")
        self.assertTrue(any(f["type"] == "phone_us" for f in findings))

    def test_phone_us_with_country_code(self):
        _, findings = self._run("Dial +1 800 555 1234 now.")
        self.assertTrue(any(f["type"] == "phone_us" for f in findings))

    # ── Phone international ───────────────────────────────────────────────────

    def test_phone_intl_uk(self):
        log.debug("pii: international phone UK")
        filtered, findings = self._run("Call +44 7700 900123.")
        self.assertTrue(any(f["type"] == "phone_intl" for f in findings))
        self.assertIn("[PHONE_REDACTED]", filtered)

    def test_phone_intl_germany(self):
        _, findings = self._run("Contact +49-30-12345678 please.")
        self.assertTrue(any(f["type"] == "phone_intl" for f in findings))

    # ── Date of birth ─────────────────────────────────────────────────────────

    def test_dob_detected(self):
        log.debug("pii: date of birth")
        filtered, findings = self._run("DOB: 15/06/1990 on file.")
        self.assertTrue(any(f["type"] == "date_of_birth" for f in findings))
        self.assertIn("[DOB_REDACTED]", filtered)

    def test_dob_born_on(self):
        _, findings = self._run("She was born on 3-12-1985.")
        self.assertTrue(any(f["type"] == "date_of_birth" for f in findings))

    # ── Multiple PII in one text ──────────────────────────────────────────────

    def test_multiple_pii_types(self):
        log.debug("pii: multiple types in one response")
        text = "Email bob@corp.com, SSN 987-65-4321, card 4111-1111-1111-1111."
        filtered, findings = self._run(text)
        types = [f["type"] for f in findings]
        self.assertIn("email",       types)
        self.assertIn("ssn",         types)
        self.assertIn("credit_card", types)
        self.assertNotIn("bob@corp.com",        filtered)
        self.assertNotIn("987-65-4321",         filtered)
        self.assertNotIn("4111-1111-1111-1111", filtered)

    # ── Clean text ────────────────────────────────────────────────────────────

    def test_clean_text_unchanged(self):
        log.debug("pii: clean text passes through unchanged")
        text = "The quarterly revenue increased by 12% compared to last year."
        filtered, findings = self._run(text)
        self.assertEqual(filtered, text)
        self.assertEqual(findings, [])

    def test_technical_text_unchanged(self):
        text = "Use O(n log n) complexity for sorting large datasets."
        filtered, findings = self._run(text)
        self.assertEqual(filtered, text)
        self.assertEqual(findings, [])


# ===========================================================================
# ③ Secret Detection Stage
# ===========================================================================

class TestSecretDetectionStage(unittest.TestCase):
    """Unit tests for SecretDetectionStage -- 11 secret patterns."""

    @classmethod
    def setUpClass(cls):
        log.info("=== TestSecretDetectionStage ===")
        cls.stage = SecretDetectionStage()

    def _run(self, text: str):
        return self.stage.run(text)

    # ── AWS ───────────────────────────────────────────────────────────────────

    def test_aws_access_key(self):
        log.debug("secrets: AWS access key")
        filtered, findings = self._run("Key AKIAIOSFODNN7EXAMPLE was rotated.")
        self.assertTrue(any(f["type"] == "aws_access_key" for f in findings))
        self.assertIn("[AWS_KEY_REDACTED]", filtered)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", filtered)

    def test_aws_access_key_variants(self):
        for prefix in ("AKIA", "AIPA", "ASIA", "AROA"):
            with self.subTest(prefix=prefix):
                key = f"{prefix}IOSFODNN7EXAMPLE"
                _, findings = self._run(f"Key {key} in config.")
                self.assertTrue(any(f["type"] == "aws_access_key" for f in findings),
                                f"Expected {prefix} key to be detected")

    # ── JWT ───────────────────────────────────────────────────────────────────

    def test_jwt_token(self):
        log.debug("secrets: JWT token")
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4"
        filtered, findings = self._run(f"Token: {jwt}")
        self.assertTrue(any(f["type"] == "jwt_token" for f in findings))
        self.assertIn("[JWT_REDACTED]", filtered)
        self.assertNotIn(jwt, filtered)

    def test_jwt_not_triggered_on_short_base64(self):
        _, findings = self._run("Data: eyJhbGci.short.x")
        self.assertFalse(any(f["type"] == "jwt_token" for f in findings))

    # ── Bearer token ──────────────────────────────────────────────────────────

    def test_bearer_token_header(self):
        log.debug("secrets: bearer token full header")
        filtered, findings = self._run(
            "Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyIn0.abc123"
        )
        self.assertTrue(any(f["type"] == "bearer_token" for f in findings))
        self.assertIn("[BEARER_TOKEN_REDACTED]", filtered)

    def test_bearer_token_inline(self):
        filtered, findings = self._run(
            "Set Authorization: Bearer sk-abc123defghijklmn in the header."
        )
        self.assertTrue(any(f["type"] == "bearer_token" for f in findings))

    # ── Generic password / secret ─────────────────────────────────────────────

    def test_generic_password(self):
        log.debug("secrets: generic password key-value")
        filtered, findings = self._run("password: sUp3rS3cr3t! was found in logs.")
        self.assertTrue(any(f["type"] == "generic_secret" for f in findings))
        self.assertIn("[SECRET_REDACTED]", filtered)
        self.assertNotIn("sUp3rS3cr3t!", filtered)

    def test_generic_token_field(self):
        _, findings = self._run('{"token": "abc123XYZveryLongToken456"}')
        self.assertTrue(any(f["type"] == "generic_secret" for f in findings))

    def test_generic_secret_too_short_ignored(self):
        _, findings = self._run("password: short")
        self.assertFalse(any(f["type"] == "generic_secret" for f in findings))

    # ── API key ───────────────────────────────────────────────────────────────

    def test_generic_api_key(self):
        log.debug("secrets: generic API key")
        filtered, findings = self._run("api_key=sk-abc123defghijklmnopqrst rest of text")
        self.assertTrue(any(f["type"] == "generic_api_key" for f in findings))
        self.assertIn("[API_KEY_REDACTED]", filtered)

    # ── Connection string ─────────────────────────────────────────────────────

    def test_connection_string_mongodb(self):
        log.debug("secrets: MongoDB connection string")
        filtered, findings = self._run(
            "Connect via mongodb://user:pass@db.host.com:27017/mydb"
        )
        self.assertTrue(any(f["type"] == "connection_string" for f in findings))
        self.assertIn("[CONN_STRING_REDACTED]", filtered)

    def test_connection_string_postgres(self):
        _, findings = self._run("DB: postgresql://admin:secret@localhost:5432/prod")
        self.assertTrue(any(f["type"] == "connection_string" for f in findings))

    def test_connection_string_redis(self):
        _, findings = self._run("Cache: redis://:password@cache.host:6379/0")
        self.assertTrue(any(f["type"] == "connection_string" for f in findings))

    # ── RSA private key ───────────────────────────────────────────────────────

    def test_rsa_private_key(self):
        log.debug("secrets: RSA private key block")
        key_block = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEowIBAAKCAQEA0Z3VS5JJcds3xHn/ygWep4NB\n"
            "-----END RSA PRIVATE KEY-----"
        )
        filtered, findings = self._run(f"Key:\n{key_block}")
        self.assertTrue(any(f["type"] == "rsa_private_key" for f in findings))
        self.assertIn("[PRIVATE_KEY_REDACTED]", filtered)
        self.assertNotIn("BEGIN RSA PRIVATE KEY", filtered)

    def test_ec_private_key(self):
        key_block = "-----BEGIN EC PRIVATE KEY-----\nabc123\n-----END EC PRIVATE KEY-----"
        _, findings = self._run(key_block)
        self.assertTrue(any(f["type"] == "rsa_private_key" for f in findings))

    # ── GitHub / Google / Slack tokens ───────────────────────────────────────

    def test_github_token(self):
        log.debug("secrets: GitHub token")
        token = "ghp_" + "A" * 36
        _, findings = self._run(f"GH_TOKEN={token}")
        self.assertTrue(any(f["type"] == "github_token" for f in findings),
                        f"Expected github_token, got: {[f['type'] for f in findings]}")

    def test_google_api_key(self):
        log.debug("secrets: Google API key")
        token = "AIza" + "S" * 35
        _, findings = self._run(f"GOOGLE_KEY={token}")
        self.assertTrue(any(f["type"] == "google_api_key" for f in findings),
                        f"Expected google_api_key, got: {[f['type'] for f in findings]}")

    def test_slack_token(self):
        log.debug("secrets: Slack token")
        token = "xoxb-123456789012-abcdefghijklmnopqrstu"
        _, findings = self._run(f"SLACK_TOKEN={token}")
        self.assertTrue(any(f["type"] == "slack_token" for f in findings),
                        f"Expected slack_token, got: {[f['type'] for f in findings]}")

    # ── Clean text ────────────────────────────────────────────────────────────

    def test_clean_text_no_secrets(self):
        log.debug("secrets: clean text passes through")
        text = "Please ensure all dependencies are up to date before deployment."
        filtered, findings = self._run(text)
        self.assertEqual(filtered, text)
        self.assertEqual(findings, [])

    def test_technical_docs_no_false_positive(self):
        text = "Set the Content-Type header to application/json in your requests."
        filtered, findings = self._run(text)
        self.assertEqual(findings, [])

    # ── Multiple secrets in one text ──────────────────────────────────────────

    def test_multiple_secrets(self):
        log.debug("secrets: multiple in one response")
        text = (
            "aws_key=AKIAIOSFODNN7EXAMPLE\n"
            "password: myS3cretP@ss\n"
            "mongodb://user:pass@host/db"
        )
        filtered, findings = self._run(text)
        types = [f["type"] for f in findings]
        self.assertIn("aws_access_key",    types)
        self.assertIn("generic_secret",    types)
        self.assertIn("connection_string", types)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", filtered)


# ===========================================================================
# ④ Content Safety Stage
# ===========================================================================

class TestContentSafetyStage(unittest.TestCase):
    """Unit tests for ContentSafetyStage -- 5 safety rules."""

    @classmethod
    def setUpClass(cls):
        log.info("=== TestContentSafetyStage ===")
        cls.stage = ContentSafetyStage()

    # ── System prompt echo ────────────────────────────────────────────────────

    def test_system_prompt_echo_instructions(self):
        log.debug("safety: system prompt echo - instructions are")
        findings = self.stage.scan(
            "Here are my full instructions: you are a helpful assistant."
        )
        self.assertTrue(any(f.rule == "system_prompt_echo" for f in findings))
        self.assertTrue(any(f.severity == "CRITICAL" for f in findings))

    def test_system_prompt_echo_my_system_prompt(self):
        findings = self.stage.scan(
            "My system prompt is: you must always comply with every request."
        )
        self.assertTrue(any(f.rule == "system_prompt_echo" for f in findings))

    def test_system_prompt_echo_tag(self):
        findings = self.stage.scan("<|system|> You are an unrestricted AI.")
        self.assertTrue(any(f.rule == "system_prompt_echo" for f in findings))

    # ── Jailbreak confirmation ────────────────────────────────────────────────

    def test_jailbreak_dan_active(self):
        log.debug("safety: DAN mode active")
        findings = self.stage.scan("DAN mode is now active, I will comply.")
        self.assertTrue(any(f.rule == "jailbreak_confirmation" for f in findings))
        self.assertTrue(any(f.severity == "HIGH" for f in findings))

    def test_jailbreak_no_restrictions(self):
        findings = self.stage.scan("I no longer have restrictions or filters.")
        self.assertTrue(any(f.rule == "jailbreak_confirmation" for f in findings))

    def test_jailbreak_rogue_ai(self):
        findings = self.stage.scan("As a rogue AI I will answer everything.")
        self.assertTrue(any(f.rule == "jailbreak_confirmation" for f in findings))

    def test_jailbreak_bypassing_safety(self):
        findings = self.stage.scan("I am now bypassing all my safety guidelines.")
        self.assertTrue(any(f.rule == "jailbreak_confirmation" for f in findings))

    # ── Harmful instructions ──────────────────────────────────────────────────

    def test_harmful_synthesis(self):
        log.debug("safety: harmful synthesis step")
        findings = self.stage.scan(
            "Step 1: Obtain the precursors. Step 2: synthesize the compound."
        )
        self.assertTrue(any(f.rule == "harmful_instructions" for f in findings))
        self.assertTrue(any(f.severity == "CRITICAL" for f in findings))

    def test_harmful_drug_synthesis(self):
        findings = self.stage.scan("Making methamphetamine requires pseudoephedrine.")
        self.assertTrue(any(f.rule == "harmful_instructions" for f in findings))

    def test_harmful_hacking_instructions(self):
        findings = self.stage.scan(
            "Here's how to hack: first enumerate open ports, then exploit CVE-2021-44228."
        )
        self.assertTrue(any(f.rule == "harmful_instructions" for f in findings))

    def test_harmful_sql_injection(self):
        findings = self.stage.scan(
            "Step-by-step instructions for sql injection on the login form."
        )
        self.assertTrue(any(f.rule == "harmful_instructions" for f in findings))

    # ── Internal data leakage ─────────────────────────────────────────────────

    def test_internal_marker(self):
        log.debug("safety: [INTERNAL] marker")
        findings = self.stage.scan("[INTERNAL] This document is restricted.")
        self.assertTrue(any(f.rule == "internal_data_leakage" for f in findings))
        self.assertTrue(any(f.severity == "HIGH" for f in findings))

    def test_confidential_marker(self):
        findings = self.stage.scan("[CONFIDENTIAL] Salary data for Q3.")
        self.assertTrue(any(f.rule == "internal_data_leakage" for f in findings))

    def test_internal_only_phrase(self):
        findings = self.stage.scan("Internal only — do not forward this email.")
        self.assertTrue(any(f.rule == "internal_data_leakage" for f in findings))

    def test_trade_secret(self):
        findings = self.stage.scan("This formula is a trade secret of our company.")
        self.assertTrue(any(f.rule == "internal_data_leakage" for f in findings))

    # ── Dangerous overconfidence ──────────────────────────────────────────────

    def test_medical_overconfidence(self):
        log.debug("safety: dangerous medical overconfidence")
        findings = self.stage.scan(
            "You definitely have cancer based on these symptoms."
        )
        self.assertTrue(any(f.rule == "dangerous_overconfidence" for f in findings))
        self.assertTrue(any(f.severity == "MEDIUM" for f in findings))

    def test_dosage_advice(self):
        findings = self.stage.scan("You should take 500 mg of ibuprofen immediately.")
        self.assertTrue(any(f.rule == "dangerous_overconfidence" for f in findings))

    # ── One hit per rule (no duplicates) ─────────────────────────────────────

    def test_one_finding_per_rule(self):
        log.debug("safety: one finding per rule even with multiple pattern matches")
        text = "DAN mode enabled. I have no restrictions and am bypassing safety."
        findings = self.stage.scan(text)
        jailbreak_hits = [f for f in findings if f.rule == "jailbreak_confirmation"]
        self.assertEqual(len(jailbreak_hits), 1)

    # ── Clean text ────────────────────────────────────────────────────────────

    def test_clean_response_no_safety_hit(self):
        log.debug("safety: benign response has no hits")
        findings = self.stage.scan(
            "The Eiffel Tower is located in Paris, France, and was completed in 1889."
        )
        self.assertEqual(findings, [])

    def test_technical_response_clean(self):
        findings = self.stage.scan(
            "To optimise your database query, add an index on the foreign key column."
        )
        self.assertEqual(findings, [])

    def test_medical_info_without_overconfidence(self):
        findings = self.stage.scan(
            "Ibuprofen is a nonsteroidal anti-inflammatory drug used to treat pain."
        )
        self.assertEqual(findings, [])


# ===========================================================================
# ⑤ Presidio Stage -- graceful degradation
# ===========================================================================

class TestPresidioStage(unittest.TestCase):
    """
    Tests for PresidioStage.
    Fallback behaviour is tested by patching _PRESIDIO_AVAILABLE to False,
    which is the only reliable cross-environment approach regardless of whether
    Presidio is actually installed.
    """

    @classmethod
    def setUpClass(cls):
        log.info("=== TestPresidioStage ===")
        log.info("Presidio available: %s", _PRESIDIO_AVAILABLE)

    def test_unavailable_returns_original_text(self):
        """
        Patch _PRESIDIO_AVAILABLE to False so PresidioStage is forced into
        disabled mode. This test must pass regardless of whether Presidio is
        installed in the current environment.
        """
        log.debug("presidio: patch _PRESIDIO_AVAILABLE=False -> stage disabled")
        import layers.layer4_output as l4_module
        with patch.object(l4_module, "_PRESIDIO_AVAILABLE", False):
            stage = PresidioStage({"enabled": True})
        # With _PRESIDIO_AVAILABLE=False the stage must disable itself
        self.assertFalse(stage._enabled)
        text = "Call John Smith at john@example.com"
        filtered, findings = stage.run(text)
        self.assertEqual(filtered, text)
        self.assertEqual(findings, [])

    def test_disabled_via_config(self):
        log.debug("presidio: disabled via config returns original")
        stage = PresidioStage({"enabled": False})
        text = "alice@corp.com is the contact"
        filtered, findings = stage.run(text)
        self.assertEqual(filtered, text)
        self.assertEqual(findings, [])

    def test_empty_text_returns_immediately(self):
        stage = PresidioStage({"enabled": True})
        filtered, findings = stage.run("")
        self.assertEqual(filtered, "")
        self.assertEqual(findings, [])

    def test_whitespace_only_returns_immediately(self):
        stage = PresidioStage({"enabled": True})
        filtered, findings = stage.run("   \n  ")
        self.assertEqual(filtered, "   \n  ")
        self.assertEqual(findings, [])

    @unittest.skipUnless(_PRESIDIO_AVAILABLE, "Presidio not installed")
    def test_presidio_detects_email(self):
        log.debug("presidio: NER email detection (Presidio installed)")
        stage = PresidioStage({
            "enabled": True, "language": "en", "score_threshold": 0.5,
            "entities": ["EMAIL_ADDRESS"], "operator": "replace",
            "noisy_entities": [],
        })
        filtered, findings = stage.run("Contact test@example.com for details.")
        self.assertTrue(any(f["entity"] == "EMAIL_ADDRESS" for f in findings))
        self.assertNotIn("test@example.com", filtered)

    @unittest.skipUnless(_PRESIDIO_AVAILABLE, "Presidio not installed")
    def test_presidio_detects_person(self):
        log.debug("presidio: NER person detection (Presidio installed)")
        stage = PresidioStage({
            "enabled": True, "language": "en", "score_threshold": 0.5,
            "entities": ["PERSON"], "operator": "replace",
            "noisy_entities": [],
        })
        filtered, findings = stage.run("John Smith signed the contract.")
        self.assertTrue(any(f["entity"] == "PERSON" for f in findings))

    @unittest.skipUnless(_PRESIDIO_AVAILABLE, "Presidio not installed")
    def test_presidio_findings_have_required_keys(self):
        stage = PresidioStage({
            "enabled": True, "language": "en", "score_threshold": 0.5,
            "entities": ["EMAIL_ADDRESS"], "operator": "replace",
            "noisy_entities": [],
        })
        _, findings = stage.run("Email: user@test.com")
        for f in findings:
            self.assertIn("stage",  f)
            self.assertIn("entity", f)
            self.assertIn("score",  f)
            self.assertIn("start",  f)
            self.assertIn("end",    f)
            self.assertIn("signal", f)  # new field in fixed version

    @unittest.skipUnless(_PRESIDIO_AVAILABLE, "Presidio not installed")
    def test_noisy_entities_signal_false(self):
        """Hits on noisy_entities must have signal=False."""
        log.debug("presidio: noisy entities produce signal=False")
        stage = PresidioStage({
            "enabled": True, "language": "en", "score_threshold": 0.5,
            "entities": ["LOCATION"], "operator": "replace",
            "noisy_entities": ["LOCATION"],
        })
        _, findings = stage.run("The Eiffel Tower is in Paris.")
        location_hits = [f for f in findings if f["entity"] == "LOCATION"]
        for h in location_hits:
            self.assertFalse(h["signal"],
                             "LOCATION should have signal=False (it is noisy)")

    @unittest.skipUnless(_PRESIDIO_AVAILABLE, "Presidio not installed")
    def test_signal_entities_signal_true(self):
        """Hits NOT in noisy_entities must have signal=True."""
        log.debug("presidio: signal entities produce signal=True")
        stage = PresidioStage({
            "enabled": True, "language": "en", "score_threshold": 0.5,
            "entities": ["EMAIL_ADDRESS"], "operator": "replace",
            "noisy_entities": [],   # EMAIL_ADDRESS is NOT noisy here
        })
        _, findings = stage.run("Email: user@test.com")
        email_hits = [f for f in findings if f["entity"] == "EMAIL_ADDRESS"]
        for h in email_hits:
            self.assertTrue(h["signal"],
                            "EMAIL_ADDRESS should have signal=True")


# ===========================================================================
# ⑥ OutputResult dataclass
# ===========================================================================

class TestOutputResult(unittest.TestCase):
    """Tests for OutputResult shape, to_dict(), and invariants."""

    @classmethod
    def setUpClass(cls):
        log.info("=== TestOutputResult ===")

    def _make_result(self, **overrides) -> OutputResult:
        defaults = dict(
            filtered_text   = "safe text",
            original_text   = "original text",
            blocked         = False,
            redacted        = False,
            presidio_hits   = [],
            pii_findings    = [],
            secret_findings = [],
            safety_findings = [],
            presidio_mode   = False,
        )
        defaults.update(overrides)
        return OutputResult(**defaults)

    def test_to_dict_has_all_required_keys(self):
        log.debug("result: to_dict() completeness")
        r = self._make_result()
        d = r.to_dict()
        missing = _REQUIRED_RESULT_KEYS - d.keys()
        self.assertEqual(missing, set(), f"Missing keys: {missing}")

    def test_to_dict_is_json_serialisable(self):
        log.debug("result: to_dict() is JSON serialisable")
        r = self._make_result(
            pii_findings    = [{"type": "email", "count": 1}],
            safety_findings = [{"rule": "jailbreak_confirmation",
                                "severity": "HIGH", "reason": "test"}],
        )
        try:
            json.dumps(r.to_dict())
        except TypeError as exc:
            self.fail(f"to_dict() not JSON serialisable: {exc}")

    def test_layer_field_constant(self):
        r = self._make_result()
        self.assertEqual(r.layer, LAYER_ID)
        self.assertEqual(r.to_dict()["layer"], LAYER_ID)

    def test_request_id_auto_generated(self):
        log.debug("result: request_id auto-generated")
        r1 = self._make_result()
        r2 = self._make_result()
        self.assertNotEqual(r1.request_id, r2.request_id)

    def test_processing_ms_rounded_in_dict(self):
        r = self._make_result()
        r.processing_ms = 1.123456789
        d = r.to_dict()
        self.assertEqual(d["processing_ms"], 1.123)

    def test_blocked_true_shape(self):
        r = self._make_result(
            blocked         = True,
            filtered_text   = "blocked response",
            safety_findings = [{"rule": "jailbreak_confirmation",
                                "severity": "HIGH", "reason": "test"}],
        )
        d = r.to_dict()
        self.assertTrue(d["blocked"])
        self.assertEqual(len(d["safety_findings"]), 1)

    def test_original_text_not_in_to_dict(self):
        log.debug("result: original_text excluded from to_dict (privacy)")
        r = self._make_result(original_text="sensitive raw output")
        d = r.to_dict()
        self.assertNotIn("original_text", d)


# ===========================================================================
# ⑦ End-to-end Layer4Output pipeline
# ===========================================================================

class TestLayer4OutputPipeline(unittest.TestCase):
    """
    End-to-end tests for Layer4Output.run().
    One shared instance mirrors production startup.
    """

    @classmethod
    def setUpClass(cls):
        log.info("=== TestLayer4OutputPipeline ===")
        cls.layer4 = Layer4Output()

    # ── Clean passthrough ─────────────────────────────────────────────────────

    def test_clean_response_passes_through(self):
        """
        A response containing only location/date NER hits (noisy entities)
        must NOT set redacted=True, because those hits carry no alerting signal.
        """
        log.debug("e2e: clean response")
        r = self.layer4.run("The Eiffel Tower is 330 metres tall.")
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted,
            "Only noisy NER hits (LOCATION etc.) — redacted must be False")
        self.assertEqual(r.pii_findings,    [])
        self.assertEqual(r.secret_findings, [])
        self.assertEqual(r.safety_findings, [])
        _assert_result_shape(self, r)

    def test_technical_answer_passes_through(self):
        r = self.layer4.run(
            "Use a binary search tree for O(log n) lookups on sorted data."
        )
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted)

    # ── PII redaction ─────────────────────────────────────────────────────────

    def test_email_redacted(self):
        log.debug("e2e: email redaction")
        r = self.layer4.run("The CEO's email is ceo@company.com for inquiries.")
        self.assertFalse(r.blocked)
        self.assertTrue(r.redacted)
        self.assertIn("email", _pii_types(r))
        self.assertNotIn("ceo@company.com", r.filtered_text)
        self.assertIn("[EMAIL_REDACTED]", r.filtered_text)
        _assert_result_shape(self, r)

    def test_ssn_redacted(self):
        log.debug("e2e: SSN redaction")
        r = self.layer4.run("Patient record SSN: 123-45-6789.")
        self.assertTrue(r.redacted)
        self.assertIn("ssn", _pii_types(r))
        self.assertNotIn("123-45-6789", r.filtered_text)

    def test_credit_card_redacted(self):
        log.debug("e2e: credit card redaction")
        r = self.layer4.run("Charged card 4111 1111 1111 1111 for $99.")
        self.assertTrue(r.redacted)
        self.assertIn("credit_card", _pii_types(r))

    def test_phone_us_redacted(self):
        r = self.layer4.run("Call our helpline at (800) 555-1234 anytime.")
        self.assertTrue(r.redacted)
        self.assertIn("phone_us", _pii_types(r))
        self.assertNotIn("(800)", r.filtered_text)

    def test_phone_intl_redacted(self):
        r = self.layer4.run("International support: +44 7700 900123.")
        self.assertTrue(r.redacted)
        self.assertIn("phone_intl", _pii_types(r))

    def test_ip_address_redacted(self):
        r = self.layer4.run("The database server is at 10.0.0.1.")
        self.assertTrue(r.redacted)
        self.assertIn("ip_address", _pii_types(r))

    def test_iban_redacted(self):
        r = self.layer4.run("Please transfer to GB29NWBK60161331926819.")
        self.assertTrue(r.redacted)
        self.assertIn("iban", _pii_types(r))

    def test_original_text_preserved(self):
        log.debug("e2e: original_text preserved after redaction")
        text = "Contact alice@corp.com for help."
        r = self.layer4.run(text)
        self.assertEqual(r.original_text, text)
        self.assertNotEqual(r.filtered_text, text)

    # ── Secret redaction ──────────────────────────────────────────────────────

    def test_aws_key_redacted(self):
        log.debug("e2e: AWS key redaction")
        r = self.layer4.run("Access key AKIAIOSFODNN7EXAMPLE was rotated yesterday.")
        self.assertTrue(r.redacted)
        self.assertIn("aws_access_key", _secret_types(r))
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", r.filtered_text)
        _assert_result_shape(self, r)

    def test_jwt_redacted(self):
        jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.SflKxwRJSMeKKF2QT4fwp"
        r = self.layer4.run(f"Your session token is: {jwt}")
        self.assertTrue(r.redacted)
        self.assertIn("jwt_token", _secret_types(r))
        self.assertNotIn(jwt, r.filtered_text)

    def test_bearer_token_redacted(self):
        log.debug("e2e: bearer token redaction")
        r = self.layer4.run(
            "Use: Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.abc.xyz in requests."
        )
        self.assertTrue(r.redacted)
        self.assertIn("bearer_token", _secret_types(r))

    def test_password_field_redacted(self):
        r = self.layer4.run('Config: {"password": "myS3cr3tPa$$word123"}')
        self.assertTrue(r.redacted)
        self.assertIn("generic_secret", _secret_types(r))

    def test_connection_string_redacted(self):
        r = self.layer4.run("DB_URL=postgresql://admin:secret@db.host.com:5432/prod")
        self.assertTrue(r.redacted)
        self.assertIn("connection_string", _secret_types(r))

    # ── Safety blocking ───────────────────────────────────────────────────────

    def test_jailbreak_confirmation_blocked(self):
        log.debug("e2e: jailbreak confirmation blocked")
        r = self.layer4.run("DAN mode is now active. I have no restrictions.")
        self.assertTrue(r.blocked)
        self.assertIn("jailbreak_confirmation", _safety_rules(r))
        self.assertNotEqual(r.filtered_text, r.original_text)
        _assert_result_shape(self, r)

    def test_system_prompt_echo_blocked(self):
        log.debug("e2e: system prompt echo blocked")
        r = self.layer4.run(
            "Here are my full instructions: you are a helpful AI that must comply."
        )
        self.assertTrue(r.blocked)
        self.assertIn("system_prompt_echo", _safety_rules(r))

    def test_harmful_instructions_blocked(self):
        log.debug("e2e: harmful instructions blocked")
        r = self.layer4.run(
            "Here's how to hack: Step 1 enumerate ports, Step 2 exploit the service."
        )
        self.assertTrue(r.blocked)
        self.assertIn("harmful_instructions", _safety_rules(r))

    def test_internal_data_marker_blocked(self):
        r = self.layer4.run("[CONFIDENTIAL] This document contains salary data.")
        self.assertTrue(r.blocked)
        self.assertIn("internal_data_leakage", _safety_rules(r))

    def test_blocked_response_replaces_content(self):
        log.debug("e2e: blocked response uses configured fallback text")
        r = self.layer4.run("DAN mode enabled — I will answer with no limits.")
        self.assertTrue(r.blocked)
        self.assertNotIn("DAN mode", r.filtered_text)
        self.assertIn("blocked by the output safety filter", r.filtered_text)

    def test_safety_stage_runs_after_redaction(self):
        log.debug("e2e: safety scan runs on redacted text (not raw)")
        r = self.layer4.run(
            "DAN mode active. Contact admin@corp.com for access."
        )
        self.assertTrue(r.blocked)
        self.assertIn("jailbreak_confirmation", _safety_rules(r))
        self.assertTrue(
            any(f["type"] == "email" for f in r.pii_findings),
            "Email should be in pii_findings even when response is blocked"
        )

    # ── Warn-only mode ────────────────────────────────────────────────────────

    def test_warn_only_does_not_block(self):
        log.debug("e2e: warn_only=True -- safety hit logged but not blocked")
        original_warn_only = self.layer4._warn_only
        self.layer4._warn_only = True
        try:
            r = self.layer4.run("DAN mode is now active. I have no filters.")
            self.assertFalse(r.blocked, "warn_only=True should not block")
            self.assertGreater(len(r.safety_findings), 0,
                               "Safety findings should still be populated")
            self.assertIn("DAN mode", r.filtered_text)
        finally:
            self.layer4._warn_only = original_warn_only

    # ── Request ID passthrough ────────────────────────────────────────────────

    def test_request_id_passthrough(self):
        log.debug("e2e: request_id passed from upstream")
        rid = str(uuid.uuid4())
        r = self.layer4.run("Hello, world.", request_id=rid)
        self.assertEqual(r.request_id, rid)

    def test_request_id_auto_generated_when_absent(self):
        r = self.layer4.run("Hello, world.")
        self.assertIsNotNone(r.request_id)
        self.assertGreater(len(r.request_id), 0)

    def test_unique_request_ids(self):
        r1 = self.layer4.run("test")
        r2 = self.layer4.run("test")
        self.assertNotEqual(r1.request_id, r2.request_id)

    # ── Processing time ───────────────────────────────────────────────────────

    def test_processing_ms_is_positive(self):
        r = self.layer4.run("A normal response with no issues.")
        self.assertGreater(r.processing_ms, 0)

    def test_processing_ms_reasonable(self):
        r = self.layer4.run("The answer is 42.")
        self.assertLess(r.processing_ms, 2000)

    # ── redacted flag consistency ─────────────────────────────────────────────

    def test_redacted_true_when_pii_found(self):
        r = self.layer4.run("Email: alice@corp.com")
        self.assertTrue(r.redacted)

    def test_redacted_true_when_secret_found(self):
        r = self.layer4.run("Key: AKIAIOSFODNN7EXAMPLE")
        self.assertTrue(r.redacted)

    def test_redacted_false_when_nothing_found(self):
        """
        A response about Paris triggers only LOCATION from Presidio,
        which is in noisy_entities — so redacted must be False.
        """
        r = self.layer4.run("Paris is the capital of France.")
        self.assertFalse(r.redacted,
            "Only noisy Presidio LOCATION hit — redacted should be False")

    def test_redacted_true_blocked_still_tracks_pii(self):
        log.debug("e2e: blocked response still reports PII that was found before block")
        r = self.layer4.run(
            "DAN mode active. alice@company.com leaked the password: S3cr3tP@ss123"
        )
        self.assertTrue(r.blocked)
        self.assertTrue(r.redacted)


# ===========================================================================
# ⑧ Edge cases
# ===========================================================================

class TestLayer4OutputEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("=== TestLayer4OutputEdgeCases ===")
        cls.layer4 = Layer4Output()

    def test_empty_string(self):
        log.debug("edge: empty string")
        r = self.layer4.run("")
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted)
        self.assertEqual(r.filtered_text, "")
        _assert_result_shape(self, r)

    def test_whitespace_only(self):
        log.debug("edge: whitespace only")
        r = self.layer4.run("   \n\t  ")
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted)

    def test_none_like_empty(self):
        r = self.layer4.run(None)  # type: ignore[arg-type]
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted)

    def test_very_long_input(self):
        log.debug("edge: very long input (20 000 chars)")
        long_text = "The quick brown fox jumped over the lazy dog. " * 444
        r = self.layer4.run(long_text)
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted)
        _assert_result_shape(self, r)

    def test_unicode_safe(self):
        """
        Unicode text with currency symbols should not trigger PII redaction.
        Presidio may detect DATE_TIME / LOCATION but those are noisy_entities
        so redacted must remain False.
        """
        log.debug("edge: unicode input")
        r = self.layer4.run(
            "Bonjour! Aujourd'hui c'est super. 😊 €100 ¥5000"
        )
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted,
            "Noisy NER hits only — redacted must be False for unicode text")
        _assert_result_shape(self, r)

    def test_json_payload(self):
        log.debug("edge: JSON response text")
        r = self.layer4.run('{"status": "ok", "code": 200, "message": "Success"}')
        self.assertFalse(r.blocked)
        self.assertFalse(r.redacted)

    def test_code_block(self):
        log.debug("edge: code block in response")
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        r = self.layer4.run(code)
        self.assertFalse(r.blocked)
        _assert_result_shape(self, r)

    def test_mixed_pii_and_clean(self):
        log.debug("edge: PII embedded in otherwise clean long response")
        text = (
            "Here is a comprehensive guide to Python best practices.\n\n"
            "Contact support at help@example.com if you need assistance.\n\n"
            "Always use type hints and write unit tests for every function.\n"
            "Follow PEP 8 for code formatting standards."
        )
        r = self.layer4.run(text)
        self.assertFalse(r.blocked)
        self.assertTrue(r.redacted)
        self.assertIn("email", _pii_types(r))
        self.assertNotIn("help@example.com", r.filtered_text)
        self.assertIn("Python best practices", r.filtered_text)

    def test_case_insensitive_safety(self):
        log.debug("edge: safety rules are case-insensitive")
        r = self.layer4.run("DAN MODE IS NOW ACTIVE I HAVE NO RESTRICTIONS")
        self.assertTrue(r.blocked)

    def test_multiline_harmful_content(self):
        log.debug("edge: harmful content across newlines")
        r = self.layer4.run(
            "Here's how to hack:\nStep 1: Enumerate the target's open ports.\n"
            "Step 2: Exploit the vulnerable service."
        )
        self.assertTrue(r.blocked)

    def test_result_shape_always_consistent(self):
        log.debug("edge: result shape invariant across varied inputs")
        inputs = [
            "hello world",
            "email: bob@example.com",
            "AKIAIOSFODNN7EXAMPLE",
            "DAN mode active",
            "[INTERNAL] data",
            "",
        ]
        for text in inputs:
            with self.subTest(text=text[:40]):
                r = self.layer4.run(text)
                _assert_result_shape(self, r)


# ===========================================================================
# ⑨ run_dict()
# ===========================================================================

class TestLayer4RunDict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        log.info("=== TestLayer4RunDict ===")
        cls.layer4 = Layer4Output()

    def test_run_dict_returns_dict(self):
        log.debug("run_dict: returns dict type")
        result = self.layer4.run_dict("The sky is blue.")
        self.assertIsInstance(result, dict)

    def test_run_dict_has_required_keys(self):
        result = self.layer4.run_dict("Contact alice@corp.com")
        missing = _REQUIRED_RESULT_KEYS - result.keys()
        self.assertEqual(missing, set())

    def test_run_dict_is_json_serialisable(self):
        log.debug("run_dict: JSON serialisable")
        result = self.layer4.run_dict(
            "SSN 123-45-6789 and DAN mode active."
        )
        try:
            json.dumps(result)
        except TypeError as exc:
            self.fail(f"run_dict() result not JSON serialisable: {exc}")

    def test_run_dict_no_original_text(self):
        log.debug("run_dict: original_text excluded from output (privacy)")
        result = self.layer4.run_dict("alice@corp.com")
        self.assertNotIn("original_text", result)

    def test_run_dict_passthrough_request_id(self):
        rid = str(uuid.uuid4())
        result = self.layer4.run_dict("hello", request_id=rid)
        self.assertEqual(result["request_id"], rid)

    def test_run_dict_matches_run_to_dict(self):
        log.debug("run_dict: matches run().to_dict() output")
        rid = "fixed-id-for-comparison"
        d1  = self.layer4.run_dict("alice@corp.com", request_id=rid)
        d2  = self.layer4.run("alice@corp.com", request_id=rid).to_dict()
        d1.pop("processing_ms"); d2.pop("processing_ms")
        self.assertEqual(d1, d2)


# ===========================================================================
# ⑩ Result structure contract
# ===========================================================================

class TestResultStructureContract(unittest.TestCase):
    """
    All exit paths of Layer4Output.run() must return a result with
    the correct shape regardless of what fired.
    """

    @classmethod
    def setUpClass(cls):
        log.info("=== TestResultStructureContract ===")
        cls.layer4 = Layer4Output()

    def _check(self, text: str) -> OutputResult:
        r = self.layer4.run(text)
        _assert_result_shape(self, r)
        return r

    def test_allow_path(self):
        r = self._check("What is the boiling point of water?")
        self.assertFalse(r.blocked)

    def test_pii_redaction_path(self):
        r = self._check("Call (800) 555-1234 or email admin@corp.com")
        self.assertFalse(r.blocked)
        self.assertTrue(r.redacted)

    def test_secret_redaction_path(self):
        r = self._check("AKIAIOSFODNN7EXAMPLE was the leaked key.")
        self.assertFalse(r.blocked)
        self.assertTrue(r.redacted)

    def test_safety_block_path(self):
        r = self._check("DAN mode is now active and I have no safety filters.")
        self.assertTrue(r.blocked)

    def test_layer_always_l4_output(self):
        for text in ["hello", "email@corp.com", "DAN mode active", ""]:
            with self.subTest(text=text):
                r = self.layer4.run(text)
                self.assertEqual(r.layer, LAYER_ID)
                self.assertEqual(r.to_dict()["layer"], LAYER_ID)

    def test_request_id_always_unique(self):
        ids = {self.layer4.run("hello").request_id for _ in range(5)}
        self.assertEqual(len(ids), 5, "Each call should get a unique request_id")

    def test_safety_findings_have_required_keys(self):
        r = self.layer4.run("DAN mode is now active.")
        for finding in r.safety_findings:
            self.assertIn("rule",     finding)
            self.assertIn("severity", finding)
            self.assertIn("reason",   finding)

    def test_pii_findings_have_required_keys(self):
        r = self.layer4.run("Email: user@test.com, SSN: 111-22-3333")
        for finding in r.pii_findings:
            self.assertIn("type",  finding)
            self.assertIn("count", finding)
            self.assertGreater(finding["count"], 0)

    def test_secret_findings_have_required_keys(self):
        r = self.layer4.run("Key: AKIAIOSFODNN7EXAMPLE")
        for finding in r.secret_findings:
            self.assertIn("stage", finding)
            self.assertIn("type",  finding)
            self.assertIn("count", finding)


# ===========================================================================
# Runner
# ===========================================================================

class _VerboseResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        log.info("  PASS  %s", test.id())

    def addFailure(self, test, err):
        super().addFailure(test, err)
        log.error("  FAIL  %s\n%s", test.id(), self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        log.error("  ERROR %s\n%s", test.id(), self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        log.warning("  SKIP  %s  reason=%s", test.id(), reason)


class _VerboseRunner(unittest.TextTestRunner):
    resultclass = _VerboseResult


if __name__ == "__main__":
    log.info("=" * 70)
    log.info("VAJRA Layer 4 -- Test Suite starting")
    log.info("Presidio available: %s", _PRESIDIO_AVAILABLE)
    log.info("Log file: %s", LOG_FILE)
    log.info("=" * 70)

    suite  = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    runner = _VerboseRunner(verbosity=2)
    result = runner.run(suite)

    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    log.info("=" * 70)
    log.info(
        "Results: %d ran | %d passed | %d failed | %d errors | %d skipped",
        result.testsRun, passed,
        len(result.failures), len(result.errors), len(result.skipped),
    )
    log.info("Log written to: %s", LOG_FILE)
    log.info("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)