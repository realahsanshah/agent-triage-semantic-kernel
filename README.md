# Agent Triage Demo — Semantic Kernel

A small multi-agent customer-support triage system built with **Microsoft Semantic
Kernel (Python)**: a `TriageAgent` routes each customer message to a
`BillingAgent` or `TechSupportAgent`, each equipped with its own tools.

This is a hands-on learning exercise, not a production system. The scope is
deliberately small — one file per concept, mock data instead of a real
backend — so the framework's building blocks stay visible.

## Architecture

```
User message
    ↓
TriageAgent (ChatCompletionAgent, no tools)
    ↓ Handoff-transfer_to_*   (auto-generated routing functions)
    ├─→ BillingAgent      (ChatCompletionAgent + BillingPlugin: check_balance, issue_refund)
    └─→ TechSupportAgent  (ChatCompletionAgent + TechSupportPlugin: check_system_status, restart_service)
    ↕ either specialist can hand back to TriageAgent if the request is out of scope
```

Run by a `HandoffOrchestration` on SK's `InProcessRuntime`.

| File | What it teaches |
|---|---|
| `plugins.py` | `@kernel_function` — turning a Python method into an LLM-callable tool |
| `middleware.py` | SK's Filter — the middleware/interceptor pattern around function calls |
| `agents.py` | `Kernel`, `ChatCompletionAgent`, `OrchestrationHandoffs` — building agents and the routing graph |
| `main.py` | `HandoffOrchestration` + `InProcessRuntime` — running multi-turn conversations |

## Why Semantic Kernel (and the Agent Framework context)

Semantic Kernel is Microsoft's original agent/orchestration SDK; **Microsoft
Agent Framework** is its successor, unifying SK's agent concepts with AutoGen's
multi-agent patterns. A lot of SK's vocabulary and mental model carries over
directly, which is the point of building this alongside the Agent Framework
version of the same scenario — it turns "architecturally transferable" from a
claim into something you can actually walk through.

### Concept glossary: SK → Agent Framework

| Semantic Kernel | What it does here | Agent Framework equivalent |
|---|---|---|
| `Kernel` | Per-agent container for the LLM connection, plugins, and filters | No direct equivalent — a `ChatAgent` just takes a `ChatClient` and a `tools=[...]` list directly |
| `@kernel_function` / Plugin class | Exposes a Python method as an LLM tool, grouped in a class | Plain functions passed straight into `tools=[...]` — no wrapper class needed |
| `ChatCompletionAgent` | One LLM + instructions + tools = one agent | `ChatAgent` |
| `OrchestrationHandoffs` + `HandoffOrchestration` | Declarative "who can hand off to whom" graph, executed by a runtime | `WorkflowBuilder` (graph/executor-based routing) |
| Kernel filter (`add_filter(FilterTypes.FUNCTION_INVOCATION, ...)`) | Middleware wrapping every function call (`await next(context)`) | A middleware function passed into `ChatAgent`/workflow construction |
| `InProcessRuntime` | Actor-style runtime executing the agent graph | No separately-surfaced concept — the runtime lives inside `WorkflowBuilder` |
| One `HandoffOrchestration.invoke()` call, driven by `human_response_function` | Where multi-turn conversation state actually lives | `AgentSession` — a first-class, explicit multi-turn object |

The biggest structural difference: SK routes multi-agent conversations through
a declarative handoff *graph* plus a runtime, while Agent Framework's
`WorkflowBuilder` is a more general graph-of-executors that a routing pattern
is one instance of, and `AgentSession` makes conversation state an explicit
object rather than something implicit inside a single `invoke()` call.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

Requires Python 3.11+ and an OpenAI API key (or adapt `agents.py` for Azure
OpenAI — see the comment in `.env.example`).

## A real gotcha I hit building this

`semantic_kernel.agents.runtime` (which `HandoffOrchestration` depends on)
imports `google.protobuf` directly, but the `semantic-kernel` wheel doesn't
declare `protobuf` as a dependency — importing `agents.py` fails with
`ModuleNotFoundError: No module named 'google'` unless you install it
separately. `requirements.txt` already pins it; worth knowing about if you
hit the same error building the Agent Framework version.

## Running it

**Scripted demo** (deterministic, no typing) — proves multi-turn context works:

```bash
python main.py
```

It plays two fixed customer turns: *"I was charged twice for my subscription"*
then *"actually, make that a partial refund instead"* — the second turn only
makes sense if the agent still remembers the first, which is what this
demonstrates. Watch the console for:
- `TriageAgent` handing off to `BillingAgent`
- `[middleware]` lines firing on both the handoff function and `issue_refund`
- the partial-refund turn being resolved without the customer repeating
  themselves

**Interactive mode** — type your own messages:

```bash
python main.py --interactive
```

Try a technical complaint (e.g. "the login page is down") to see it route to
`TechSupportAgent` and call `check_system_status` / `restart_service` instead.

## Talking points (what each piece does)

- **`ChatCompletionAgent`** — one agent = one LLM connection + instructions +
  its own `Kernel` (so its tool schema doesn't leak into other agents').
- **`OrchestrationHandoffs`** — a graph, not code: each `.add()`/`.add_many()`
  call becomes an auto-generated `Handoff-transfer_to_X` tool the model can
  call when it decides the conversation belongs elsewhere.
- **`HandoffOrchestration` + `InProcessRuntime`** — the runtime executes that
  graph. A single `.invoke()` call is the whole multi-turn session; it keeps
  pulling more input via `human_response_function` until an agent calls the
  built-in "task complete" function.
- **The filter in `middleware.py`** — SK's middleware primitive. Registered
  per-`Kernel`, wraps every function call with your own before/after code.

## What I'd add with more time

- A real backend instead of in-memory mock data
- Compare `GroupChatOrchestration` (deliberation between agents) and
  `MagenticOrchestration` (planner-driven) against this `HandoffOrchestration`
  approach for the same scenario
- Persistence/checkpointing of conversation state across process restarts
- OpenTelemetry integration (SK has built-in hooks for this) instead of
  print-based logging
- Unit tests that mock the chat completion service so the plugin/handoff
  logic can be tested without live API calls
