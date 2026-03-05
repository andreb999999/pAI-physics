import time
import os

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import anthropic

_client: anthropic.Anthropic = None  # type: ignore
# _client: anthropic.AnthropicBedrock = None  # type: ignore

ANTHROPIC_TIMEOUT_EXCEPTIONS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.APIStatusError,
)


@once
def _setup_anthropic_client():
    global _client
    _client = anthropic.Anthropic(max_retries=0)
    # _client = anthropic.AnthropicBedrock(max_retries=0)


def _normalize_tool_spec(func_spec: FunctionSpec | dict) -> tuple[str, dict, dict]:
    if isinstance(func_spec, FunctionSpec):
        return (
            func_spec.name,
            func_spec.as_anthropic_tool_dict,
            func_spec.anthropic_tool_choice_dict,
        )

    if isinstance(func_spec, dict):
        tool_name = func_spec.get("name")
        tool_desc = func_spec.get("description", "")
        input_schema = func_spec.get("json_schema", func_spec.get("parameters", {}))
        if not tool_name:
            raise ValueError(f"Invalid Anthropic tool spec, missing name: {func_spec}")
        return (
            tool_name,
            {
                "name": tool_name,
                "description": tool_desc,
                "input_schema": input_schema,
            },
            {
                "type": "tool",
                "name": tool_name,
            },
        )

    raise TypeError(f"Unsupported func_spec type for Anthropic backend: {type(func_spec)}")


def query(
    system_message: str | None,
    user_message: str | list | None,
    func_spec: FunctionSpec | dict | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_anthropic_client()

    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore
    if "max_tokens" not in filtered_kwargs:
        filtered_kwargs["max_tokens"] = 8192  # default for Claude models

    tool_name = None
    if func_spec is not None:
        tool_name, tool_spec, tool_choice = _normalize_tool_spec(func_spec)
        filtered_kwargs["tools"] = [tool_spec]
        filtered_kwargs["tool_choice"] = tool_choice

    # Anthropic doesn't allow not having a user messages
    # if we only have system msg -> use it as user msg
    if system_message is not None and user_message is None:
        system_message, user_message = user_message, system_message

    # Anthropic passes the system messages as a separate argument
    if system_message is not None:
        filtered_kwargs["system"] = system_message

    messages = opt_messages_to_list(None, user_message)

    t0 = time.time()
    message = backoff_create(
        _client.messages.create,
        ANTHROPIC_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0
    print(filtered_kwargs)

    if func_spec is not None:
        tool_blocks = [
            block for block in message.content if getattr(block, "type", None) == "tool_use"
        ]
        if not tool_blocks:
            raise ValueError(f"Claude returned no tool_use block for tool {tool_name}.")
        selected_block = next(
            (block for block in tool_blocks if getattr(block, "name", None) == tool_name),
            tool_blocks[0],
        )
        output: OutputType = selected_block.input
    else:
        text_blocks = [
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ]
        if not text_blocks:
            raise ValueError("Claude returned no text block.")
        output = "\n".join(text_blocks).strip()

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens

    info = {
        "stop_reason": message.stop_reason,
    }

    return output, req_time, in_tokens, out_tokens, info
