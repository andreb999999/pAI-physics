from __future__ import annotations
from typing import Optional, Type, Any
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict


class TalkToUserInput(BaseModel):
    message: str = Field(description="The message to display to the user (question, instruction, etc.)")


class TalkToUser(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "talk_to_user"
    description: str = """
    This tool displays a message from the LLM to the user (e.g., a question, statement, or instruction),
    waits for the user's reply, and returns the response.
    """
    args_schema: Type[BaseModel] = TalkToUserInput

    def _run(self, message: str) -> str:
        print(f"[LLM]: {message}")
        user_response = input("[User]: ")
        return user_response

    async def _arun(self, **kwargs: Any) -> str:
        raise NotImplementedError
