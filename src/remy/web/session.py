"""
Web Session Manager — LangGraph agent for web/desktop channel.

Same brain, same tools, same personality as Telegram. Single-user local app.
Supports text, voice (audio blob), and file/image uploads via multimodal HumanMessage.
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from google import genai

from remy.config.settings import settings
from remy.core.agent_tools import brain
from remy.core.brain_tools import generate_session_summary

# Lazy — langgraph/langchain_core load only on first chat message, not at startup
def _load_invoke_agent():
    from remy.core.agent import invoke_agent
    return invoke_agent

def _invoke_agent_stream():
    from remy.core.agent import invoke_agent_stream
    return invoke_agent_stream


async def invoke_agent(*args, **kwargs):
    return await _load_invoke_agent()(*args, **kwargs)

logger = logging.getLogger("WebSession")

# ============== MULTIMODAL CONSTANTS ==============

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

SUPPORTED_MIME_TYPES = {
    # Images
    "image/jpeg", "image/png", "image/gif", "image/webp",
    # Audio
    "audio/webm", "audio/wav", "audio/mp3", "audio/mpeg",
    "audio/mp4", "audio/ogg", "audio/flac",
    # Documents
    "application/pdf",
    # Text
    "text/plain", "text/csv",
}


@dataclass
class WebSession:
    """Single-user web session state."""

    session_id: str
    history: list = field(default_factory=list)
    session_log: list = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)


class WebSessionManager:
    """Manages a single user's web session with LangGraph agent."""

    def __init__(self):
        self.client = None
        self.readonly = True
        self.refresh_credentials()
        self.session: WebSession | None = None

    def refresh_credentials(self) -> None:
        """Refresh API-key dependent client state after settings changes."""
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.readonly = not api_key

        if api_key:
            # Client kept only for session summary generation
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = None
            logger.warning("No API key — running in readonly mode (brain access only, chat disabled)")

    def get_or_create_session(self) -> WebSession:
        """Get existing session or create a new one."""
        if self.session is not None:
            self.session.last_activity = time.time()
            return self.session

        self.session = WebSession(
            session_id=str(uuid.uuid4()),
            last_activity=time.time(),
        )
        logger.info(f"New web session: {self.session.session_id[:8]}...")
        return self.session

    async def close_session(self):
        """Close session: generate summary, end brain session, clear state."""
        if self.session is None:
            return

        session = self.session
        self.session = None  # Prevent re-entrant close
        
        # Save session history to JSON — skip empty sessions (no messages sent)
        user_turns = [e for e in session.session_log if e.get("type") in ("user_text", "user_voice")]
        if user_turns:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                filename = f"{timestamp}_{session.session_id}.json"
                history_dir = settings.DATA_DIR / "history"
                history_dir.mkdir(parents=True, exist_ok=True)

                filepath = history_dir / filename

                data = {
                    "session_id": session.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "log": session.session_log
                }

                from remy.core.file_utils import atomic_write
                atomic_write(filepath, json.dumps(data, indent=2, ensure_ascii=False))

                logger.info(f"Saved session history to {filepath}")
            except Exception as e:
                logger.error(f"Failed to save session history: {e}")
        else:
            logger.debug("Empty session — skipping history save")

        if self.client:
            try:
                await asyncio.wait_for(
                    generate_session_summary(
                        self.client, session.session_log, session.session_id
                    ),
                    timeout=15.0,
                )
            except asyncio.CancelledError:
                logger.info("Session summary cancelled during shutdown")
            except asyncio.TimeoutError:
                logger.warning("Session summary timed out (15s)")
            except Exception as e:
                logger.warning(f"Session summary failed: {e}")

        try:
            from remy.core.agent_tools import brain_lock

            def _end_session_locked():
                with brain_lock:
                    brain.end_session(session.session_id)

            await asyncio.to_thread(_end_session_locked)
        except Exception as e:
            logger.warning(f"end_session failed: {e}")

        logger.info(f"Web session closed: {session.session_id[:8]}...")
        self.session = None

    # ============== TEXT RESPOND ==============

    async def gemini_respond(self, user_text: str) -> str:
        """Send user text message through LangGraph agent.

        Returns the final text response.
        """
        if self.readonly:
            return ("No API key configured. Chat is disabled. "
                    "Go to Settings to add your Gemini API key, then try again.")

        session = self.get_or_create_session()

        from remy.core.logging_config import log_context
        with log_context(session_id=session.session_id, channel="desktop"):
            session.session_log.append({"type": "user_text", "text": user_text[:200]})

            response_text, new_history, new_log = await invoke_agent(
                user_message=user_text,
                session_id=session.session_id,
                channel="desktop",
                session_log=session.session_log,
                history=session.history,
            )

            session.history = new_history
            session.session_log = new_log
            return response_text

    # ============== MULTIMODAL RESPOND ==============

    async def gemini_respond_multimodal(
        self,
        text: str | None = None,
        attachments: list[dict] | None = None,
        is_voice: bool = False,
    ) -> dict:
        """Send multimodal user message (text + audio/files) through LangGraph agent.

        Args:
            text: Optional text message.
            attachments: List of {"mime_type": str, "data": bytes} dicts.
            is_voice: If True, the primary attachment is voice audio.

        Returns:
            {"response": str, "input_transcript": str | None}
        """
        if self.readonly:
            return {"response": "No API key configured. Go to Settings to add your Gemini API key.", "input_transcript": None}

        from langchain_core.messages import HumanMessage

        session = self.get_or_create_session()

        content_parts = []

        # Validate and add attachments
        for att in (attachments or []):
            mime = att["mime_type"]
            data = att["data"]

            if mime not in SUPPORTED_MIME_TYPES:
                return {"response": f"Unsupported file type: {mime}", "input_transcript": None}

            if len(data) > MAX_FILE_SIZE:
                return {"response": "File too large (max 20MB).", "input_transcript": None}

            # Convert to base64 data URL for LangChain
            b64_data = base64.b64encode(data).decode("utf-8")

            if mime.startswith("image/"):
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64_data}"},
                })
            else:
                # Audio, PDF, text files — use media type
                content_parts.append({
                    "type": "media",
                    "mime_type": mime,
                    "data": b64_data,
                })

            if is_voice:
                session.session_log.append({"type": "user_voice", "mime": mime, "size": len(data)})
            else:
                session.session_log.append({"type": "user_file", "mime": mime, "size": len(data)})

        # Add text part
        if text:
            content_parts.append({"type": "text", "text": text})
            session.session_log.append({"type": "user_text", "text": text[:200]})
        elif is_voice and not text:
            content_parts.insert(0, {
                "type": "text",
                "text": "The user sent a voice message. Listen to it, understand what they said, "
                        "and respond naturally. Do NOT start with a transcription — just respond to their message.",
            })

        if not content_parts:
            return {"response": "Empty message.", "input_transcript": None}

        user_msg = HumanMessage(content=content_parts)

        response_text, new_history, new_log = await invoke_agent(
            user_message=user_msg,
            session_id=session.session_id,
            channel="desktop",
            session_log=session.session_log,
            history=session.history,
        )

        session.history = new_history
        session.session_log = new_log

        return {"response": response_text, "input_transcript": None}


    async def gemini_respond_stream(self, user_text: str, model_override: str | None = None):
        """Send user text and yield streaming events.

        Args:
            user_text: The user's message.
            model_override: If set, temporarily use this model instead of settings.SUMMARY_MODEL.

        Yields:
            dict: Event from invoke_agent_stream
        """
        if self.readonly:
            yield {"type": "token", "content": "No API key configured."}
            yield {"type": "final", "text": "No API key configured."}
            return

        from remy.core.agent import invoke_agent_stream
        from remy.config.settings import settings as _settings

        session = self.get_or_create_session()
        session.session_log.append({"type": "user_text", "text": user_text[:200]})

        original_model = None
        if model_override and model_override != _settings.SUMMARY_MODEL:
            original_model = _settings.SUMMARY_MODEL
            _settings.SUMMARY_MODEL = model_override

        try:
            async for event in invoke_agent_stream(
                user_message=user_text,
                session_id=session.session_id,
                channel="desktop",
                session_log=session.session_log,
                history=session.history,
            ):
                if event["type"] == "final":
                    session.history = event["messages"]
                    if event.get("session_log") is not None:
                        session.session_log = event["session_log"]
                    else:
                        session.session_log.append({
                            "type": "model_response",
                            "text": event["text"][:200],
                        })
                yield event
        finally:
            if original_model is not None:
                _settings.SUMMARY_MODEL = original_model
