import json
import logging
from datetime import datetime

from pydantic import ValidationError

from dualmemory.models import CandidateFact, Episode, Message
from dualmemory.providers import LLM

logger = logging.getLogger(__name__)


# The extraction prompt. Real wording gets iterated later (DESIGN.md: prompt
# versions are hashed/frozen). For now: ask for strict JSON in a known shape.
EXTRACT_PROMPT = """Extract memory from the conversation below.
Return ONLY a JSON object with two keys:
  "episodes": list of {{"text": str, "event_time": ISO8601}}
  "facts":    list of {{"subject": str, "text": str, "confidence": float, "event_time": ISO8601}}

Conversation:
{conversation}
"""


def _strip_fences(raw: str) -> str:
    """LLMs often wrap JSON in ```json ... ``` fences. Strip them."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        s = s.removeprefix("json")
    return s.strip("` \n")


class Extractor:
    """Turns messages into validated Episodes + CandidateFacts using an LLM.

    The LLM proposes; pydantic validates. Anything that fails validation is
    skipped and logged, so one malformed generation can't crash ingestion."""

    def __init__(self, llm: LLM, max_retries: int = 1) -> None:
        self.llm = llm
        self.max_retries = max_retries

    def extract(
        self, messages: list[Message], next_episode_id: int
    ) -> tuple[list[Episode], list[CandidateFact]]:
        prompt = self._build_prompt(messages)

        parsed = self._call_and_parse(prompt)
        if parsed is None:
            return [], []   # gave up after retries; skip this batch

        episodes = self._to_episodes(parsed, messages, next_episode_id)
        candidates = self._to_candidates(parsed)
        return episodes, candidates

    # --- prompt ----------------------------------------------------------
    @staticmethod
    def _build_prompt(messages: list[Message]) -> str:
        convo = "\n".join(f"[{m.timestamp.isoformat()}] {m.role}: {m.content}"
                          for m in messages)
        return EXTRACT_PROMPT.format(conversation=convo)

    # --- call + parse with retry ----------------------------------------
    def _call_and_parse(self, prompt: str) -> dict | None:
        for attempt in range(self.max_retries + 1):
            raw = self.llm.complete(prompt)
            try:
                return json.loads(_strip_fences(raw))
            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("extraction parse failed (attempt %d): %s",
                               attempt + 1, exc)
        return None   # all attempts failed

    # --- build typed objects, validating each ---------------------------
    @staticmethod
    def _to_episodes(
        parsed: dict, messages: list[Message], next_id: int
    ) -> list[Episode]:
        src_ids = [m.id for m in messages]
        episodes: list[Episode] = []
        for item in parsed.get("episodes", []):
            try:
                ep = Episode(
                    id=next_id,
                    conversation_id=messages[0].conversation_id,
                    text=item["text"],
                    event_time=datetime.fromisoformat(item["event_time"]),
                    source_message_ids=src_ids,
                )
                episodes.append(ep)
                next_id += 1
            except (KeyError, ValueError, ValidationError) as exc:
                logger.warning("skipping malformed episode: %s", exc)
        return episodes

    @staticmethod
    def _to_candidates(parsed: dict) -> list[CandidateFact]:
        candidates: list[CandidateFact] = []
        for item in parsed.get("facts", []):
            try:
                candidates.append(CandidateFact(
                    subject=item["subject"],
                    text=item["text"],
                    confidence=item.get("confidence", 1.0),
                    event_time=datetime.fromisoformat(item["event_time"]),
                ))
            except (KeyError, ValueError, ValidationError) as exc:
                logger.warning("skipping malformed candidate fact: %s", exc)
        return candidates