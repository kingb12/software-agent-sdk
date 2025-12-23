"""MessageUser tool implementation for sending a user message."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field
from rich.text import Text

from openhands.sdk.logger.logger import get_logger
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)
from openhands.sdk.tool.tool import ToolExecutor


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import (
        ConversationState,
    )


class MessageUserAction(Action):
    """Schema for sending a direct message to the user."""

    message: str = Field(description="message to send to the user.")

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("MessageUser with message:\n", style="bold blue")
        content.append(self.message)
        return content


class MessageUserObservation(Observation):
    @property
    def visualize(self) -> Text:
        return Text()


TOOL_DESCRIPTION = """Sends a direct message to the user to continue this conversation

Use this tool when:
- You need to ask the user a question, seek their confirmation, or otherwise communicate with them
"""


class MessageUserExecutor(ToolExecutor):
    def __call__(
        self,
        action: MessageUserAction,
        conversation: "BaseConversation | None" = None,
    ) -> MessageUserObservation:
        # Mark conversation as finished when this tool is called
        if conversation is not None:
            # Import here to avoid hard runtime dependency at module import time
            conversation.state.execution_status = "finished"
        else:
            logger = get_logger(__name__)
            logger.error("No conversation provided to message_user exectutor!")
        # Return a simple confirmation that the message was sent
        return MessageUserObservation.from_text(text="message sent")


class MessageUserTool(ToolDefinition[MessageUserAction, MessageUserObservation]):
    """Tool for messaging the user and finishing the conversation."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        **params,
    ) -> Sequence["MessageUserTool"]:
        if params:
            raise ValueError("MessageUserTool doesn't accept parameters")
        enhanced_description = f"{TOOL_DESCRIPTION}"

        return [
            cls(
                description=enhanced_description,
                action_type=MessageUserAction,
                observation_type=MessageUserObservation,
                annotations=ToolAnnotations(
                    title="message_user",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=MessageUserExecutor(),
            )
        ]


# Automatically register the tool when this module is imported
register_tool(MessageUserTool.name, MessageUserTool)
