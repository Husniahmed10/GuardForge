import logfire
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses openai/gpt-oss-120b via Groq for intent classification.
    """
    global _rails

    try:
        guard_llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model="openai/gpt-oss-120b",
            temperature=0
        )

        config = RailsConfig.from_content(
            colang_content=COLANG_CONTENT,
            yaml_content=YAML_CONTENT
        )

        _rails = LLMRails(config, llm=guard_llm)
        logfire.info("🛡️ NeMo Guardrails initialised (openai/gpt-oss-120b).")
    except Exception as e:
        logfire.error(f"❌ Guardrails failed to initialise — all queries will bypass the gate: {e}")
        raise  # fail fast so startup logs clearly show the problem
    
    


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        result = _rails.generate(messages=[{"role": "user", "content": message}])

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
