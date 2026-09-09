"""Entry point: wires the agents into a HandoffOrchestration and runs it.

Two SK concepts land here:

- HandoffOrchestration + InProcessRuntime: the runtime that actually executes
  the agent graph. You .invoke() it once with a starting task; internally it
  keeps calling `human_response_function` for more user turns and
  `agent_response_callback` for every agent message, until an agent decides
  the task is complete. This single invoke() call *is* the multi-turn
  session - SK doesn't have a separate "AgentSession" object the way Agent
  Framework does; the conversation state lives inside the orchestration run.

- agent_response_callback: an observer that fires on every agent message
  (including internal tool-call/tool-result messages). This is a different,
  coarser hook than the middleware.py filter: the filter fires per function
  call, this fires per chat message.

Run modes:
  python main.py                -> scripted demo (deterministic, no typing)
  python main.py --interactive  -> you type the customer's messages yourself
"""

import argparse
import asyncio
import itertools

from dotenv import load_dotenv
from semantic_kernel.agents import HandoffOrchestration
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.contents import AuthorRole, ChatMessageContent, FunctionCallContent, FunctionResultContent

from agents import get_agents_and_handoffs

# The PRD's own example: same customer, second message only makes sense if the
# agent remembers the first one said "charged twice".
SCRIPTED_TURNS = [
    "Hi, I'm customer cust_1 and I was charged twice for my subscription.",
    "Actually, don't do a full refund - just make that a partial refund of $10 instead.",
]
CLOSING_MESSAGE = "No, that's everything - thank you, goodbye!"
MAX_SCRIPTED_EXCHANGES = 6  # safety valve in case an agent keeps asking questions


def agent_response_callback(message: ChatMessageContent) -> None:
    print(f"{message.name}: {message.content}")
    for item in message.items:
        if isinstance(item, FunctionCallContent):
            print(f"  -> calling '{item.name}' with {item.arguments}")
        if isinstance(item, FunctionResultContent):
            print(f"  <- result from '{item.name}': {item.result}")


def make_scripted_human_response_function():
    """Feeds SCRIPTED_TURNS one at a time, then CLOSING_MESSAGE, then a hard stop."""
    turns = itertools.chain(SCRIPTED_TURNS, itertools.repeat(CLOSING_MESSAGE))
    count = 0

    def human_response_function() -> ChatMessageContent:
        nonlocal count
        count += 1
        if count > MAX_SCRIPTED_EXCHANGES:
            text = CLOSING_MESSAGE
        else:
            text = next(turns)
        print(f"User: {text}")
        return ChatMessageContent(role=AuthorRole.USER, content=text)

    return human_response_function


def interactive_human_response_function() -> ChatMessageContent:
    text = input("User: ")
    return ChatMessageContent(role=AuthorRole.USER, content=text)


async def main(interactive: bool) -> None:
    load_dotenv()

    agent_list, handoffs = get_agents_and_handoffs()
    human_response_function = (
        interactive_human_response_function if interactive else make_scripted_human_response_function()
    )

    orchestration = HandoffOrchestration(
        members=agent_list,
        handoffs=handoffs,
        agent_response_callback=agent_response_callback,
        human_response_function=human_response_function,
    )

    runtime = InProcessRuntime()
    runtime.start()

    result = await orchestration.invoke(
        task="Greet the customer who is reaching out for support.",
        runtime=runtime,
    )
    summary = await result.get()
    print(f"\nTask summary: {summary}")

    await runtime.stop_when_idle()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Kernel billing/tech-support triage demo")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Type your own messages instead of running the scripted 2-turn demo",
    )
    args = parser.parse_args()
    asyncio.run(main(args.interactive))
