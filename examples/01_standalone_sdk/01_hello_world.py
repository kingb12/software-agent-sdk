import os

from openhands.sdk import LLM, Agent, Conversation, Event, LLMConvertibleEvent
from openhands.tools.preset.dialogue import get_dialogue_tools


llm = LLM(
    model=os.getenv("LLM_MODEL", "gpt-5-nano"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None),
)

tools = get_dialogue_tools()
agent = Agent(llm=llm, tools=tools)

cwd = os.getcwd()

llm_messages = []


def conversation_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())


conversation = Conversation(
    agent=agent, workspace=cwd, callbacks=[conversation_callback]
)

conversation.send_message("Use the message user tool to ask me my favorite color")
conversation.run()
print("All done!")

print("=" * 100)
print("Conversation finished. Got the following LLM messages:")
for i, message in enumerate(llm_messages):
    print(f"Message {i}: {str(message)[:200]}")
