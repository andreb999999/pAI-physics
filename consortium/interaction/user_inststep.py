"""
UserInstructionStep — packages live-steering instructions for graph state injection.

Used by callback_tools.py to package a live-steering instruction from the user
socket into a form that can be injected into ResearchState.messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.messages import HumanMessage


@dataclass
class UserInstructionStep:
    """
    Represents a live instruction injected by the user via the interrupt socket.

    Attributes:
        user_instruction: The instruction text typed by the user.
        is_new_task:       True → treat as a brand-new task; False → modify current task.
    """

    user_instruction: str
    is_new_task: bool = False

    def to_messages(self) -> List[HumanMessage]:
        prefix = "New task from user" if self.is_new_task else "Additional instruction from the user"
        return [HumanMessage(content=f"{prefix}:\n{self.user_instruction}")]
