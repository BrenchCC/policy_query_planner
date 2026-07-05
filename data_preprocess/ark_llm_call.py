import os
import logging
from typing import Any

from volcenginesdkarkruntime import Ark

logger = logging.getLogger(__name__)

SOURCE_URL = (
    "https://raw.githubusercontent.com/BrenchCC/OpenAI_API_utils/"
    "158bcbb94d50db6ac92e271fb96cae326dbaf642/ark_llm_call.py"
)


def _ensure_ark_api_key() -> None:
    """Ensure the Ark API key is available in the environment.

    Raises:
        RuntimeError: If the API key is missing.
    """
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("Missing LLM_API_KEY in environment")


def call_llm_on_volcengine(
    input_query: str,
    end_point: str,
    system_prompt: str | None = None,
    stream: bool = False,
    reasoning_option: str | None = None
) -> tuple[str | None, str, int | str, int | str]:
    """Call a Volcengine Ark chat-completion endpoint.

    Args:
        input_query: User prompt sent to the endpoint.
        end_point: Ark endpoint or model identifier.
        system_prompt: Optional system prompt.
        stream: Whether to request a streaming response.
        reasoning_option: Optional Ark thinking mode.

    Returns:
        Reasoning text, response text, prompt tokens, and completion tokens.
    """
    _ensure_ark_api_key()
    client = Ark(
        base_url = os.environ.get("LLM_API_BASE_URL"),
        api_key = os.environ.get("LLM_API_KEY")
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": input_query}]
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    request_options: dict[str, Any] = {
        "model": end_point,
        "messages": messages,
        "timeout": 300,
        "stream": stream
    }
    if reasoning_option:
        request_options["extra_body"] = {"thinking": {"type": reasoning_option}}

    try:
        completion = client.chat.completions.create(**request_options)
        if stream:
            chunks = []
            for token in completion:
                if token.choices and token.choices[0].delta.content:
                    chunks.append(token.choices[0].delta.content)
            return None, "".join(chunks), "", ""

        result = completion.choices[0].message.content
        reasoning_content = getattr(completion.choices[0].message, "reasoning_content", "")
        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", "")
        completion_tokens = getattr(usage, "completion_tokens", "")
        if not isinstance(result, str):
            return reasoning_content, "dummy_result", prompt_tokens, completion_tokens
        return reasoning_content, result, prompt_tokens, completion_tokens
    except Exception as error:
        logger.exception("Ark request failed: %s", error)
        return None, "dummy_result", "", ""

