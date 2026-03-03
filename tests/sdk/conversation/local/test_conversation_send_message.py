from pydantic import SecretStr

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation import Conversation, LocalConversation
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.conversation.types import (
    ConversationCallbackType,
    ConversationTokenCallbackType,
)
from openhands.sdk.event.llm_convertible import MessageEvent, SystemPromptEvent
from openhands.sdk.llm import LLM, Message, TextContent


class SendMessageDummyAgent(AgentBase):
    def __init__(self):
        llm = LLM(
            model="gpt-4o-mini", api_key=SecretStr("test-key"), usage_id="test-llm"
        )
        super().__init__(llm=llm, tools=[])

    def init_state(
        self, state: ConversationState, on_event: ConversationCallbackType
    ) -> None:
        event = SystemPromptEvent(
            source="agent", system_prompt=TextContent(text="dummy"), tools=[]
        )
        on_event(event)

    def step(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        on_event(
            MessageEvent(
                source="agent",
                llm_message=Message(role="assistant", content=[TextContent(text="ok")]),
            )
        )


def test_send_message_with_string_creates_correct_message():
    """Test that send_message with string creates the correct Message structure."""
    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    test_text = "Hello, world!"
    conversation.send_message(test_text)

    # Should have system prompt + user message
    assert len(conversation.state.events) == 2

    # Check the user message event
    user_event = conversation.state.events[-1]
    assert isinstance(user_event, MessageEvent)
    assert user_event.source == "user"

    # Check the message structure
    message = user_event.llm_message
    assert message.role == "user"
    assert len(message.content) == 1
    assert isinstance(message.content[0], TextContent)
    assert message.content[0].text == test_text


def test_send_message_string_equivalent_to_message_object():
    """Test that send_message with string produces the same result as with Message object."""  # noqa: E501
    agent1 = SendMessageDummyAgent()
    agent2 = SendMessageDummyAgent()

    conversation1 = Conversation(agent=agent1, workspace="/tmp/workspace")
    conversation2 = Conversation(agent=agent2, workspace="/tmp/workspace")

    test_text = "Test message"

    # Use send_message with string
    conversation1.send_message(test_text)

    # Use send_message with Message object
    message = Message(role="user", content=[TextContent(text=test_text)])
    conversation2.send_message(message)

    # Both should have the same number of events
    assert len(conversation1.state.events) == len(conversation2.state.events)

    # The user message events should be equivalent
    user_event1 = conversation1.state.events[-1]
    user_event2 = conversation2.state.events[-1]

    assert isinstance(user_event1, MessageEvent)
    assert isinstance(user_event2, MessageEvent)

    assert user_event1.source == user_event2.source
    assert user_event1.llm_message.role == user_event2.llm_message.role
    assert isinstance(user_event1.llm_message.content[0], TextContent)
    assert isinstance(user_event2.llm_message.content[0], TextContent)
    assert (
        user_event1.llm_message.content[0].text
        == user_event2.llm_message.content[0].text
    )


def test_send_message_with_empty_string():
    """Test that send_message works with empty string."""
    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    conversation.send_message("")

    # Should have system prompt + user message
    assert len(conversation.state.events) == 2

    user_event = conversation.state.events[-1]
    assert isinstance(user_event, MessageEvent)
    assert isinstance(user_event.llm_message.content[0], TextContent)
    assert user_event.llm_message.content[0].text == ""


def test_send_message_with_multiline_string():
    """Test that send_message works with multiline strings."""
    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    test_text = "Line 1\nLine 2\nLine 3"
    conversation.send_message(test_text)

    # Should have system prompt + user message
    assert len(conversation.state.events) == 2

    user_event = conversation.state.events[-1]
    assert isinstance(user_event, MessageEvent)
    assert isinstance(user_event.llm_message.content[0], TextContent)
    assert user_event.llm_message.content[0].text == test_text


def test_send_message_with_message_object():
    """Test that send_message works with Message objects (existing functionality)."""
    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    test_text = "Test message"
    message = Message(role="user", content=[TextContent(text=test_text)])
    conversation.send_message(message)

    # Should have system prompt + user message
    assert len(conversation.state.events) == 2

    user_event = conversation.state.events[-1]
    assert isinstance(user_event, MessageEvent)
    assert user_event.source == "user"
    assert user_event.llm_message.role == "user"
    assert len(user_event.llm_message.content) == 1
    assert isinstance(user_event.llm_message.content[0], TextContent)
    assert user_event.llm_message.content[0].text == test_text


def test_send_message_with_assistant_role_creates_agent_source_event():
    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    text = "Here is the question I have for you."
    message = Message(role="assistant", content=[TextContent(text=text)])
    conversation.send_message(message)

    # Should have system prompt + injected assistant message
    assert len(conversation.state.events) == 2

    injected_event = conversation.state.events[-1]
    assert isinstance(injected_event, MessageEvent)
    # source should be "agent" (not "user") for assistant-role messages
    assert injected_event.source == "agent"
    assert injected_event.llm_message.role == "assistant"
    assert isinstance(injected_event.llm_message.content[0], TextContent)
    assert injected_event.llm_message.content[0].text == text


def test_send_message_assistant_does_not_reset_execution_status():
    """Injecting an assistant message must NOT flip execution_status FINISHED -> IDLE.

    This is the key invariant for the Devstral workaround: after message_user fires
    and the conversation is FINISHED, the client injects an assistant-role echo of
    the message_user content before sending the real user reply.  That injection
    must not restart the agent loop prematurely.
    """
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    # Simulate the conversation being in FINISHED state (e.g. after message_user ran)
    conversation._state.execution_status = ConversationExecutionStatus.FINISHED

    assistant_msg = Message(
        role="assistant", content=[TextContent(text="What would you like to do?")]
    )
    conversation.send_message(assistant_msg)

    # Status must stay FINISHED — do not restart the agent
    assert conversation.state.execution_status == ConversationExecutionStatus.FINISHED


def test_send_message_user_still_resets_execution_status_from_finished():
    """A normal user message must still flip FINISHED -> IDLE as before."""
    from openhands.sdk.conversation.state import ConversationExecutionStatus

    agent = SendMessageDummyAgent()
    conversation = Conversation(agent=agent, workspace="/tmp/workspace")

    conversation._state.execution_status = ConversationExecutionStatus.FINISHED

    conversation.send_message("Hello again")

    assert conversation.state.execution_status == ConversationExecutionStatus.IDLE
