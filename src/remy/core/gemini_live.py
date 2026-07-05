"""
Gemini Live Audio — real-time voice conversation with Aura Cognitive memory.

Uses Gemini 2.5 Flash Native Audio for zero-latency voice.
Brain (Aura Cognitive) connected directly — no MCP overhead.

Usage:
    remy
    remy --log-level DEBUG
"""

import asyncio
import logging
import os
import traceback
import uuid

from google import genai
from google.genai import types

from remy.config.settings import settings
from remy.core.agent_tools import brain
from remy.core.brain_tools import (
    execute_tool,
    build_system_instruction,
    get_registry,
    generate_session_summary,
)

logger = logging.getLogger("GeminiLive")
pyaudio = None

# Audio constants (pyaudio.paInt16 == 8 — avoids importing pyaudio at module level)
FORMAT = 8
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


# ============== SESSION ==============

class GeminiLiveSession:
    """Real-time voice conversation with Gemini + direct brain access."""

    def __init__(self):
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env or environment")

        self.client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )
        self.model = settings.GEMINI_MODEL
        self.voice = settings.GEMINI_VOICE

        self.session_id = str(uuid.uuid4())
        self._session_log: list[dict] = []
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.audio_stream = None
        global pyaudio
        if pyaudio is None:
            import pyaudio as _pyaudio
            pyaudio = _pyaudio
        self.pya = pyaudio.PyAudio()

    async def _listen_audio(self):
        """Capture microphone audio and send to Gemini."""
        mic_info = self.pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        logger.info("Microphone active: %s", mic_info.get("name", "default"))

        while True:
            data = await asyncio.to_thread(
                self.audio_stream.read, CHUNK_SIZE, exception_on_overflow=False
            )
            if self.out_queue is not None:
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def _send_audio(self):
        """Forward queued audio to Gemini session."""
        while True:
            msg = await self.out_queue.get()
            if self.session is not None:
                await self.session.send(input=msg)

    async def _receive_and_play(self):
        """Receive from Gemini: handle audio, text, and tool calls."""
        play_stream = await asyncio.to_thread(
            self.pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )

        while True:
            if self.session is None:
                await asyncio.sleep(0.1)
                continue

            turn = self.session.receive()
            async for response in turn:
                # Audio data → play
                if data := response.data:
                    await asyncio.to_thread(play_stream.write, data)

                # Text → print
                elif text := response.text:
                    print(text, end="", flush=True)

                # Tool call → execute and respond
                elif response.tool_call:
                    for fc in response.tool_call.function_calls:
                        logger.info(f"Tool call: {fc.name}({fc.args})")
                        fc_args = dict(fc.args)
                        result = await asyncio.to_thread(
                            execute_tool, fc.name, fc_args, self.session_id
                        )
                        logger.info(f"Tool result: {result[:200]}")
                        await self.session.send_tool_response(
                            function_responses=types.FunctionResponse(
                                name=fc.name,
                                response={"result": result},
                                id=fc.id,
                            )
                        )
                        # Track for session summary
                        self._session_log.append({
                            "type": "tool_call",
                            "tool": fc.name,
                            "args": {k: str(v)[:100] for k, v in fc_args.items()},
                            "result": result[:200],
                        })

    async def _proactive_loop(self):
        """Periodically check brain for insights and surface them to Gemini."""
        await asyncio.sleep(60)  # Let session settle before first check

        interval = settings.PROACTIVE_INTERVAL_SEC

        while True:
            try:
                from remy.core.agent_tools import brain_lock
                def _insights_locked():
                    with brain_lock:
                        return brain.insights()
                insights = await asyncio.to_thread(_insights_locked)

                if insights:
                    important = [i for i in insights if i["type"] in (
                        "decay_risk", "conflict", "hot_topic"
                    )]

                    if important and self.session:
                        summary = self._format_insights(important[:3])
                        if summary:
                            await self.session.send(
                                input=f"[INTERNAL BRAIN INSIGHT — mention naturally if relevant]: {summary}",
                                end_of_turn=True,
                            )
                            logger.info(f"Proactive insight sent: {summary[:100]}")

            except Exception as e:
                logger.warning(f"Proactive loop error: {e}")

            await asyncio.sleep(interval)

    @staticmethod
    def _format_insights(insights: list[dict]) -> str:
        """Format insights into natural language for Gemini."""
        parts = []
        for ins in insights:
            if ins["type"] == "decay_risk":
                records = ins["details"].get("records", [])
                names = [r["content"][:50] for r in records[:3]]
                if names:
                    parts.append(f"Some important memories are fading: {', '.join(names)}")
            elif ins["type"] == "conflict":
                pairs = ins["details"].get("pairs", [])
                if pairs:
                    p = pairs[0]
                    parts.append(f"Possible contradiction: '{p['content_a'][:40]}' vs '{p['content_b'][:40]}'")
            elif ins["type"] == "hot_topic":
                topics = ins["details"].get("topics", [])
                names = [t["tag"] for t in topics[:3]]
                if names:
                    parts.append(f"Active topics: {', '.join(names)}")
        return "; ".join(parts)

    async def _text_input(self):
        """Handle text input from console."""
        while True:
            text = await asyncio.to_thread(input, "message > ")
            if text.lower() == "q":
                break
            if self.session is not None:
                await self.session.send(input=text or ".", end_of_turn=True)
                self._session_log.append({"type": "user_text", "text": text})

    async def run(self):
        """Main loop: connect to Gemini, stream audio, handle tools."""
        registry = get_registry()
        all_tools = registry.get_all_declarations()
        tools_config = registry.get_tools_config()

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voice
                    )
                )
            ),
            system_instruction=types.Content(
                parts=[types.Part(text=build_system_instruction(channel="voice"))]
            ),
            tools=tools_config,
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
        )

        from remy.core.agent_tools import brain_lock
        with brain_lock:
            brain_count = brain.count()
        sandbox_count = len(registry.manifest.get_approved_tools())
        tool_names = [t.name for t in all_tools]

        print("=" * 50)
        print("FAMILY HISTORIAN — GEMINI LIVE AUDIO")
        print(f"Model: {self.model}")
        print(f"Voice: {self.voice}")
        print(f"Brain: {settings.AURA_BRAIN_PATH} ({brain_count} records)")
        print(f"Tools: {len(tool_names)} ({len(tool_names) - sandbox_count} core + {sandbox_count} sandbox)")
        print(f"  {', '.join(tool_names)}")
        print(f"Session: {self.session_id[:8]}...")
        print(f"Proactive: every {settings.PROACTIVE_INTERVAL_SEC}s")
        print("Speak naturally or type 'q' to quit.")
        print("=" * 50)

        logger.info("Connecting to Gemini Live API...")

        try:
            async with (
                self.client.aio.live.connect(
                    model=self.model, config=config
                ) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                text_task = tg.create_task(self._text_input())
                tg.create_task(self._send_audio())
                tg.create_task(self._listen_audio())
                tg.create_task(self._receive_and_play())
                tg.create_task(self._proactive_loop())

                await text_task
                raise asyncio.CancelledError("User exit")

        except asyncio.CancelledError:
            logger.info("Session ended by user")
        except ExceptionGroup as eg:
            if self.audio_stream is not None:
                self.audio_stream.close()
            traceback.print_exception(eg)
        finally:
            # Generate and store session summary
            try:
                await generate_session_summary(
                    self.client, self._session_log, self.session_id
                )
            except Exception as e:
                logger.warning(f"Session summary failed: {e}")

            # Consolidate co-activation from this session
            try:
                from remy.core.agent_tools import brain_lock
                with brain_lock:
                    result = brain.end_session(self.session_id)
                logger.info(f"Session ended: {result}")
            except Exception as e:
                logger.warning(f"end_session failed: {e}")

            # Run memory maintenance
            try:
                with brain_lock:
                    decayed, archived = brain.decay()
                if decayed or archived:
                    logger.info(f"Decay: {decayed} decayed, {archived} archived")
            except Exception as e:
                logger.warning(f"Decay failed: {e}")

            self.pya.terminate()


async def run_gemini_live():
    """Entry point for Gemini Live mode."""
    session = GeminiLiveSession()
    await session.run()
