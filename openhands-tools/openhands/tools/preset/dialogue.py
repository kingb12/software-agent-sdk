"""Dialogue preset: default tools plus MessageUser.

Provides a parallel preset to the default one, adding the MessageUser tool.
This preset registers MessageUser and returns a preset list of tools.
Agent construction is left to the caller.
"""

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import Tool


logger = get_logger(__name__)


def register_dialogue_tools(enable_browser: bool = True) -> None:
    """Register the dialogue preset tools (default + MessageUser)."""
    # Default tools (registered via their imports, or factory if required)
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.message_user import MessageUserTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    logger.debug("Tool: %s available.", TerminalTool.name)
    logger.debug("Tool: %s available.", FileEditorTool.name)
    logger.debug("Tool: %s available.", TaskTrackerTool.name)
    logger.debug("Tool: %s available.", MessageUserTool.name)
    logger.debug("Tool: %s registered for dialogue preset.", MessageUserTool.name)


def get_dialogue_tools() -> list[Tool]:
    """Return Tool specs for the dialogue preset (default + MessageUser)."""
    register_dialogue_tools()

    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.message_user import MessageUserTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal import TerminalTool

    tools = [
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
        Tool(name=MessageUserTool.name),
    ]
    return tools


# Caller constructs the Agent; this preset only provides tools.
