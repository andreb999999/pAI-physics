"""
VLM utility functions — vision-language model calls via raw Anthropic/OpenAI clients.

These helpers are used by toolkits that need direct multimodal API access
(e.g. VLMDocumentAnalysisTool). For LangGraph agent model creation, use
utils.create_model() instead.
"""

import base64
import logging
import os
from typing import Any, Dict, List, Optional, Union

import anthropic
import backoff
import openai

logger = logging.getLogger(__name__)

from .models import get_provider
from .token_usage_tracker import record_token_usage


def _record_response_token_usage(response: Any, model_id: str, source: str) -> None:
    """Record usage from raw OpenAI/Anthropic responses into run-scoped token totals."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        else:
            prompt_tokens = int(
                getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)) or 0
            )
            completion_tokens = int(
                getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)) or 0
            )

        if prompt_tokens == 0 and completion_tokens == 0:
            return

        record_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source=source,
            model_id=model_id,
        )

        # Also record USD cost to BudgetManager (VLM uses raw OpenAI/Anthropic
        # clients, not litellm.completion, so the monkey-patch doesn't cover these).
        try:
            from .budget import get_global_budget_manager
            mgr = get_global_budget_manager()
            if mgr and (prompt_tokens or completion_tokens):
                mgr.record_usage(
                    model_id=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        except Exception:
            pass  # Never break VLM calls for budget tracking errors

    except Exception:
        logger.debug("Failed to record VLM token usage", exc_info=True)


def encode_image_to_base64(image_data: Union[str, bytes, List[bytes]]) -> str:
    """Encode image data to base64 string for VLM usage."""
    if isinstance(image_data, str):
        with open(image_data, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image_data, list):
        return base64.b64encode(image_data[0]).decode("utf-8")
    elif isinstance(image_data, bytes):
        return base64.b64encode(image_data).decode("utf-8")
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")


@backoff.on_exception(
    backoff.expo,
    (openai.RateLimitError, openai.APITimeoutError, anthropic.RateLimitError),
    max_tries=5,
)
def get_response_from_vlm(
    prompt: str,
    images: List[str],
    client,
    model: str,
    system_message: str = "",
    print_debug: bool = False,
    msg_history: Optional[List[Dict]] = None,
    temperature: float = 0.75,
) -> tuple[str, List[Dict]]:
    """
    Get a response from a Vision-Language Model with image inputs.

    Args:
        prompt:         Text prompt for the VLM.
        images:         List of image file paths.
        client:         OpenAI or Anthropic client instance.
        model:          Vision-capable model name.
        system_message: System message for the conversation.
        print_debug:    Print full message history when True.
        msg_history:    Prior conversation turns.
        temperature:    Sampling temperature.

    Returns:
        (response_content, updated_message_history)
    """
    if msg_history is None:
        msg_history = []

    content: List[Dict] = [{"type": "text", "text": prompt}]
    for image_path in images:
        try:
            b64 = encode_image_to_base64(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        except Exception as e:
            logger.warning("Failed to encode image %s: %s", image_path, e)

    new_msg_history = msg_history + [{"role": "user", "content": content}]

    provider = get_provider(model)

    if provider == "openai":
        messages: List[Dict] = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.extend(new_msg_history)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            n=1,
            seed=0,
            timeout=180,
        )
        _record_response_token_usage(response, model, "vlm")
        content_response = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content_response}]

    elif provider == "anthropic":
        anthropic_content: List[Dict] = [{"type": "text", "text": prompt}]
        for image_path in images:
            try:
                b64 = encode_image_to_base64(image_path)
                anthropic_content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                })
            except Exception as e:
                logger.warning("Failed to encode image %s: %s", image_path, e)

        response = client.messages.create(
            model=model,
            system=system_message,
            messages=[{"role": "user", "content": anthropic_content}],
            max_tokens=4096,
            timeout=180,
        )
        _record_response_token_usage(response, model, "vlm")
        text_blocks = [
            block.text
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "text"
        ]
        content_response = "\n".join(text_blocks).strip()
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content_response}]

    else:
        raise ValueError(
            f"VLM model '{model}' (provider={provider}) not supported. "
            "Supported providers: openai, anthropic."
        )

    if print_debug:
        logger.debug("VLM START")
        for j, msg in enumerate(new_msg_history):
            logger.debug('%d, %s: %s', j, msg["role"], msg["content"])
        logger.debug("VLM response: %s", content_response)
        logger.debug("VLM END")

    return content_response, new_msg_history


def create_vlm_client(model: str = "gpt-4o-2024-05-13"):
    """
    Create a VLM client for vision tasks.

    Returns:
        (client, model_name) tuple.
    """
    provider = get_provider(model)
    if provider == "openai":
        return openai.OpenAI(), model
    elif provider == "anthropic":
        return anthropic.Anthropic(), model
    else:
        logger.warning("Model '%s' not supported for VLM. Defaulting to gpt-4o-2024-05-13.", model)
        return openai.OpenAI(), "gpt-4o-2024-05-13"
