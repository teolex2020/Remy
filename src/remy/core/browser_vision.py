"""
Browser Vision — Screenshot analysis via Gemini vision model.

Takes a PNG screenshot and returns structured JSON describing the page:
elements, forms, selectors, and an optional answer to the user's question.
"""

import base64
import json
import logging

from remy.config.settings import settings

logger = logging.getLogger("BrowserVision")

VISION_PROMPT_TEMPLATE = """Analyze this web page screenshot.
URL: {page_url}
{question_line}
Visible text excerpt (first 2000 chars):
{page_text}

Respond ONLY with valid JSON (no markdown fences):
{{
  "description": "What the page shows (1-2 sentences)",
  "elements": [
    {{"type": "button|link|input|select|checkbox|form", "text": "visible label/text", "selector": "UNIQUE CSS selector", "purpose": "what it does"}}
  ],
  "forms": [
    {{"name": "form purpose", "fields": [{{"label": "field label", "selector": "UNIQUE CSS selector", "type": "text|email|password|tel|number|date|select|checkbox|radio", "required": true, "placeholder": "placeholder text if visible"}}], "submit_selector": "CSS selector for submit"}}
  ],
  "answer": "Direct answer to the user's question if applicable, otherwise null",
  "suggested_actions": ["1-3 short suggestions for what the agent could do next on this page"],
  "page_state": "normal|error|captcha|loading|success"
}}

CRITICAL selector rules (follow strictly):
1. Every selector MUST uniquely identify ONE element on the page
2. Selector priority: #id > [name="..."] > form [name="..."] > label association > [placeholder="..."] > nth-child
3. NEVER use bare type selectors like input[type="email"] or input[type="password"] — they match multiple elements
4. For forms with multiple inputs, use CONTEXTUAL selectors:
   - #email or [name="email"] (best)
   - form#loginForm input[name="email"] (scoped to form)
   - label:has-text("Email") + input (adjacent to label)
   - input[placeholder="Enter your email"] (by placeholder)
   - form input:nth-of-type(1) (by position — last resort)
5. For forms, list ALL visible fields including hidden ones that affect submission
6. If the page shows an error, validation message, or captcha, mention it in description AND set page_state
7. List up to 15 most important interactive elements
8. Keep description concise"""


async def analyze_screenshot(
    screenshot_png: bytes,
    question: str = "",
    page_url: str = "",
    page_text: str = "",
) -> dict:
    """Analyze a screenshot using the vision model.

    Args:
        screenshot_png: PNG image bytes.
        question: Optional question about the page.
        page_url: Current page URL for context.
        page_text: Extracted visible text (truncated).

    Returns:
        Dict with description, elements, forms, answer, suggested_actions.
    """
    from google import genai as google_genai
    from google.genai import types

    question_line = f"User question: {question}" if question else ""
    prompt = VISION_PROMPT_TEMPLATE.format(
        page_url=page_url,
        question_line=question_line,
        page_text=page_text[:2000],
    )

    client = google_genai.Client(api_key=settings.GEMINI_API_KEY)

    b64_image = base64.b64encode(screenshot_png).decode("utf-8")

    try:
        response = client.models.generate_content(
            model=settings.BROWSER_VISION_MODEL,
            contents=[
                types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=screenshot_png,
                            )
                        ),
                        types.Part(text=prompt),
                    ]
                )
            ],
        )

        raw_text = response.text.strip()

        # Strip markdown code fences if model wraps response
        if raw_text.startswith("```"):
            # Remove ```json\n...\n```
            lines = raw_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_text = "\n".join(lines)

        result = json.loads(raw_text)
        logger.info("Vision analysis: %d elements, %d forms",
                     len(result.get("elements", [])),
                     len(result.get("forms", [])))
        return result

    except json.JSONDecodeError:
        logger.warning("Vision model returned non-JSON, using raw text")
        return {
            "description": raw_text[:500],
            "elements": [],
            "forms": [],
            "answer": raw_text[:1000] if question else None,
            "suggested_actions": [],
        }
    except Exception as e:
        logger.error("Vision analysis failed: %s", e)
        return {
            "description": f"Vision analysis failed: {e}",
            "elements": [],
            "forms": [],
            "answer": None,
            "suggested_actions": [],
            "error": str(e),
        }
