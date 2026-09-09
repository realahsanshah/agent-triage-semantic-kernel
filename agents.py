"""Builds the triage/billing/tech agents and the handoff graph between them.

Concepts introduced here:

- Kernel: SK's per-agent container for services (the LLM connection) and
  plugins (tools). We give each agent its own Kernel rather than sharing one,
  so BillingAgent's tools never leak into TechSupportAgent's function-calling
  schema. Agent Framework has no equivalent object - a ChatAgent just takes a
  ChatClient and a tools list directly.

- ChatCompletionAgent: one LLM bound to instructions + a kernel (services +
  tools). This is SK's ChatAgent equivalent.

- OrchestrationHandoffs: a declarative "who can hand off to whom, and why"
  graph. SK turns each entry into an auto-generated Handoff-transfer_to_X
  kernel function that the model can call - which is why our logging filter
  also fires on routing decisions, not just tool calls.
"""

import os

from semantic_kernel import Kernel
from semantic_kernel.agents import Agent, ChatCompletionAgent, HandoffOrchestration, OrchestrationHandoffs
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.filters import FilterTypes

from middleware import logging_filter
from plugins import BillingPlugin, TechSupportPlugin


def build_kernel(plugin: object | None = None) -> Kernel:
    """Create a Kernel with the shared OpenAI chat service, the logging filter,
    and (optionally) one plugin registered on it."""
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            ai_model_id=os.environ.get("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini"),
            api_key=os.environ["OPENAI_API_KEY"],
        )
    )
    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, logging_filter)
    if plugin is not None:
        kernel.add_plugin(plugin, plugin_name=type(plugin).__name__)
    return kernel


def get_agents_and_handoffs() -> tuple[list[Agent], OrchestrationHandoffs]:
    """Build the three agents and the handoff relationships between them."""

    triage_agent = ChatCompletionAgent(
        name="TriageAgent",
        description="Routes customer support requests to the right specialist.",
        instructions=(
            "You are the first point of contact for customer support. "
            "Greet the customer, understand their issue, and hand off to "
            "BillingAgent for billing/refund/payment issues or TechSupportAgent "
            "for outages/service issues. Do not try to solve the issue yourself."
        ),
        kernel=build_kernel(),
    )

    billing_agent = ChatCompletionAgent(
        name="BillingAgent",
        description="Handles billing questions, charges, and refunds.",
        instructions=(
            "You handle billing issues: checking balances and issuing refunds. "
            "Use your tools rather than guessing numbers. If the request is not "
            "billing-related, hand back to TriageAgent."
        ),
        kernel=build_kernel(BillingPlugin()),
    )

    tech_agent = ChatCompletionAgent(
        name="TechSupportAgent",
        description="Handles technical/service outage issues.",
        instructions=(
            "You handle technical support: checking system status and restarting "
            "services. Use your tools rather than guessing. If the request is not "
            "technical, hand back to TriageAgent."
        ),
        kernel=build_kernel(TechSupportPlugin()),
    )

    handoffs = (
        OrchestrationHandoffs()
        .add_many(
            source_agent=triage_agent.name,
            target_agents={
                billing_agent.name: "Transfer here for billing, charges, or refund requests",
                tech_agent.name: "Transfer here for outages or technical/service issues",
            },
        )
        .add(
            source_agent=billing_agent.name,
            target_agent=triage_agent.name,
            description="Transfer back if the request is not billing related",
        )
        .add(
            source_agent=tech_agent.name,
            target_agent=triage_agent.name,
            description="Transfer back if the request is not technical",
        )
    )

    return [triage_agent, billing_agent, tech_agent], handoffs
