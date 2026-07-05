"""
Remy - System Settings
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional, get_args, get_origin
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


def _app_root() -> Path:
    """Return the application root directory.

    In a PyInstaller frozen exe: directory containing the .exe file.
    In normal Python: project root (4 levels above this file).
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parents[3]


def _src_root() -> Path:
    """Return the src/remy package root (for bundled package-data files).

    In frozen mode: same as app root (PyInstaller copies package data there).
    In normal Python: src/remy directory.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parents[1]


class Settings(BaseSettings):
    """Main system settings"""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ============== PATHS ==============
    BASE_DIR: Path = Field(default_factory=_app_root)
    DATA_DIR: Path = Field(default_factory=lambda: _app_root() / "data")

    # ============== AURA COGNITIVE (episodic memory) ==============
    AURA_BRAIN_PATH: Path = Field(default_factory=lambda: _app_root() / "data" / "brain")

    # ============== SPECIALIST BASE (cognitive module) ==============
    # Path to a BasePack JSON file to load on startup (e.g. security-ops-v1.json).
    # On first run: loads JSON, runs maintenance, seals a .cog snapshot.
    # On subsequent runs: restores snapshot directly — no JSON parsing.
    REMI_BASE_PACK: Optional[str] = Field(default=None)
    # base_id and version must match the values inside the BasePack JSON.
    REMI_BASE_PACK_ID: Optional[str] = Field(default=None)
    REMI_BASE_PACK_VERSION: Optional[str] = Field(default=None)

    # ============== AURA MEMORY (semantic knowledge base) ==============
    AURA_MEMORY_PATH: Path = Field(default_factory=lambda: _app_root() / "data" / "knowledge")
    AURA_MEMORY_ENABLED: bool = Field(default=True)

    # ============== SANDBOX ==============
    SANDBOX_DIR: Path = Field(default_factory=lambda: _app_root() / "data" / "sandbox")
    SANDBOX_TOOLS_DIR: Path = Field(default_factory=lambda: _src_root() / "sandbox" / "tools")
    AURA_WHEEL_PATH: Path = Field(
        default_factory=lambda: _app_root() / "vendor" / "aura_memory-1.5.4-cp312-cp312-win_amd64.whl"
    )

    # ============== PROACTIVE ==============
    PROACTIVE_INTERVAL_SEC: int = Field(default=300)

    # ============== GEMINI ==============
    GEMINI_API_KEY: Optional[str] = Field(default=None)
    GEMINI_MODEL: str = Field(default="gemini-3.1-flash-live-preview")
    GEMINI_VOICE: str = Field(default="Zephyr")
    SUMMARY_MODEL: str = Field(default="gemini-3-flash-preview")
    WEB_SEARCH_MODEL: str = Field(default="gemini-2.5-flash")

    # ============== FALLBACK ==============
    FALLBACK_MODELS: list[str] = Field(default_factory=list)
    OPENAI_API_KEY: Optional[str] = Field(default=None)

    # ============== ADAPTIVE MODEL ROUTER ==============
    # Uses structural task pressure, model registry pricing, and lived outcome
    # memory. It does not route by prompt keywords.
    MODEL_ROUTER_ENABLED: bool = Field(default=True)

    # ============== PRIVACY ==============
    # Tokenize personal data at the LLM boundary and restore it locally.
    PII_SHIELD_ENABLED: bool = Field(default=True)

    # ============== OPENROUTER ==============
    OPENROUTER_API_KEY: Optional[str] = Field(default=None)

    OLLAMA_BASE_URL: str = Field(default="http://127.0.0.1:11434")

    @field_validator("FALLBACK_MODELS", mode="before")
    @classmethod
    def parse_fallback_models(cls, v):
        if isinstance(v, str):
            return [m.strip() for m in v.split(",") if m.strip()]
        return v or []

    # ============== TELEGRAM ==============
    TELEGRAM_BOT_TOKEN: Optional[str] = Field(default=None)
    PROACTIVE_CHAT_ID: Optional[int] = Field(default=None)
    TELEGRAM_ALLOWED_CHAT_IDS: list[int] = Field(default_factory=list)
    PRIMARY_REMOTE_SURFACE: str = Field(default="telegram")
    TELEGRAM_OPERATOR_ALERT_MIN_LEVEL: str = Field(
        default="warning"
    )

    @field_validator("TELEGRAM_ALLOWED_CHAT_IDS", mode="before")
    @classmethod
    def parse_allowed_chat_ids(cls, v):
        if isinstance(v, str):
            return [int(x.strip()) for x in v.split(",") if x.strip()]
        if isinstance(v, int):
            return [v]
        return v or []

    @field_validator("TELEGRAM_OPERATOR_ALERT_MIN_LEVEL", mode="before")
    @classmethod
    def normalize_operator_alert_level(cls, v):
        allowed = {"info", "warning", "critical"}
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in allowed:
                return normalized
        return "warning"

    # ============== WEB GUI ==============
    WEB_HOST: str = Field(default="127.0.0.1")
    WEB_PORT: int = Field(default=8080)
    WEB_ENABLED: bool = Field(default=True)
    TELEGRAM_SUPPRESS_WHEN_WEB_ENABLED: bool = Field(
        default=True
    )

    @field_validator("WEB_HOST", mode="before")
    @classmethod
    def force_local_web_host(cls, v):
        """Keep the unauthenticated desktop web server local-only."""
        host = str(v or "127.0.0.1").strip().lower()
        if host in {"127.0.0.1", "localhost", "::1"}:
            return host
        return "127.0.0.1"

    # ============== AUTONOMY ==============
    AUTONOMY_ENABLED: bool = Field(default=False)
    AUTONOMY_V3: bool = Field(default=True)
    AUTONOMY_CYCLE_INTERVAL_SEC: int = Field(default=120)
    AUTONOMY_DAILY_TOKEN_LIMIT: int = Field(default=100_000)
    AUTONOMY_HOURLY_TOKEN_LIMIT: int = Field(default=20_000)
    AUTONOMY_SESSION_TOKEN_LIMIT: int = Field(default=500_000)
    AUTONOMY_AUTO_APPROVE_SANDBOX: bool = Field(default=False)
    AUTONOMY_TELEGRAM_NOTIFICATIONS: bool = Field(default=True)
    AUTONOMY_MAX_ACTIONS_PER_HOUR: int = Field(default=20)
    AUTONOMY_QUIET_HOURS_START: int = Field(default=23)
    AUTONOMY_QUIET_HOURS_END: int = Field(default=7)
    AUTONOMY_MAX_SESSION_MINUTES: int = Field(default=30)
    AUTONOMY_ALLOWED_READ_PATHS: list = Field(default_factory=list)
    PACKS_DISABLED: list[str] = Field(default_factory=list)

    @field_validator("PACKS_DISABLED", mode="before")
    @classmethod
    def parse_disabled_packs(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v or []

    # ============== WORKERS (multi-agent delegation) ==============
    WORKER_TIMEOUT_SEC: int = Field(default=60)
    WORKER_MAX_PARALLEL: int = Field(default=3)
    WORKER_MAX_TOOL_ITERATIONS: int = Field(default=8)

    # ============== BROWSER (Playwright) ==============
    BROWSER_ENABLED: bool = Field(default=True)
    BROWSER_VISION_MODEL: str = Field(default="gemini-flash-lite-latest")
    BROWSER_HEADLESS: bool = Field(default=True)
    BROWSER_IDLE_TIMEOUT_SEC: int = Field(default=300)
    BROWSER_DAILY_ACTION_LIMIT: int = Field(default=200)
    BROWSER_PAGE_TIMEOUT_MS: int = Field(default=30000)
    BROWSER_BACKEND: str = Field(default="playwright")
    PINCHTAB_ENABLED: bool = Field(default=False)
    PINCHTAB_BASE_URL: str = Field(default="http://127.0.0.1:8941")
    PINCHTAB_TIMEOUT_SEC: int = Field(default=20)
    PINCHTAB_PROFILE_ID: str = Field(default="")
    PINCHTAB_AUTOSTART: bool = Field(default=True)
    PINCHTAB_BINARY_PATH: str = Field(default="")
    PINCHTAB_INSTALL_DIR: Path = Field(
        default_factory=lambda: _app_root() / "data" / "tools" / "pinchtab",
    )
    PINCHTAB_BOOTSTRAP_SOURCE: str = Field(default="")
    PINCHTAB_RELEASE_URL: str = Field(default="")
    PINCHTAB_COMMAND: str = Field(
        default="pinchtab serve --host 127.0.0.1 --port 8941",
    )

    # ============== AUDIT TRAIL ==============
    AUDIT_LOG_DIR: Path = Field(default_factory=lambda: _app_root() / "data" / "audit_logs")
    CRITICAL_TOOLS: list[str] = Field(default_factory=list)

    @field_validator("CRITICAL_TOOLS", mode="before")
    @classmethod
    def parse_critical_tools(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v or []

    # ============== HUMAN-IN-THE-LOOP APPROVAL QUEUE ==============
    APPROVAL_QUEUE_ENABLED: bool = Field(default=True)
    APPROVAL_TIMEOUT_SEC: int = Field(default=120)
    GUIDANCE_QUEUE_ENABLED: bool = Field(default=True)
    GUIDANCE_TIMEOUT_SEC: int = Field(default=120)

    # ============== MULTI-MODEL REVIEW ==============
    REVIEW_MODEL: str = Field(default="gemini-flash-lite-latest")
    REVIEW_ENABLED: bool = Field(default=True)
    AUTONOMY_DAILY_COST_LIMIT_USD: float = Field(default=5.0)

    # ============== WEB PUSH ==============
    VAPID_PUBLIC_KEY: str = Field(default="")
    VAPID_PRIVATE_KEY: str = Field(default="")
    VAPID_CLAIM_EMAIL: str = Field(default="mailto:user@example.com")

    # ============== SMTP (Email integration) ==============
    SMTP_HOST: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: Optional[str] = Field(default=None)
    SMTP_PASSWORD: Optional[str] = Field(default=None)
    SMTP_FROM: str = Field(default="")

    # ============== PROACTIVE SESSIONS ==============
    AUTONOMY_PROACTIVE_SESSIONS_ENABLED: bool = Field(default=True)
    AUTONOMY_PROACTIVE_MAX_PER_DAY: int = Field(default=3)
    AUTONOMY_PROACTIVE_MIN_INTERVAL_SEC: int = Field(default=7200)



# Global settings instance
settings = Settings()
RUNTIME_SETTINGS_FILE = settings.DATA_DIR / "runtime_settings.json"


def load_runtime_settings() -> dict:
    """Load runtime-managed setting overrides from data/runtime_settings.json."""
    try:
        if RUNTIME_SETTINGS_FILE.exists():
            data = json.loads(RUNTIME_SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _serialize_runtime_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def _set_env_mirror(key: str, value) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = str(value)


def _coerce_runtime_value(field_annotation, current_value, raw_value):
    if raw_value is None:
        return None

    origin = get_origin(field_annotation)
    args = [arg for arg in get_args(field_annotation) if arg is not type(None)]
    annotation = args[0] if args else field_annotation

    if annotation is bool or isinstance(current_value, bool):
        if isinstance(raw_value, str):
            return raw_value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw_value)
    if annotation is int or (isinstance(current_value, int) and not isinstance(current_value, bool)):
        return int(raw_value)
    if annotation is float or isinstance(current_value, float):
        return float(raw_value)
    if annotation is Path or isinstance(current_value, Path):
        return Path(raw_value)
    if origin is list or isinstance(current_value, list):
        if isinstance(raw_value, str):
            return [item.strip() for item in raw_value.split(",") if item.strip()]
        return raw_value
    return raw_value


def apply_runtime_settings(target: Settings = settings) -> dict:
    """Apply persisted runtime overrides to the in-memory settings object."""
    applied = {}
    for key, raw_value in load_runtime_settings().items():
        if not hasattr(target, key):
            continue
        current_value = getattr(target, key)
        field = target.__class__.model_fields.get(key)
        annotation = field.annotation if field else type(current_value)
        coerced = _coerce_runtime_value(annotation, current_value, raw_value)
        setattr(target, key, coerced)
        _set_env_mirror(key, coerced)
        applied[key] = coerced
    return applied


def set_runtime_setting(key: str, value, target: Settings = settings):
    """Persist a runtime-managed setting override without touching .env."""
    data = load_runtime_settings()
    if value is None:
        data.pop(key, None)
        _set_env_mirror(key, None)
    else:
        data[key] = _serialize_runtime_value(value)
    RUNTIME_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_SETTINGS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if hasattr(target, key):
        current_value = getattr(target, key)
        field = target.__class__.model_fields.get(key)
        annotation = field.annotation if field else type(current_value)
        coerced = _coerce_runtime_value(annotation, current_value, value)
        setattr(target, key, coerced)
        _set_env_mirror(key, coerced)
        return coerced
    return value


apply_runtime_settings()

