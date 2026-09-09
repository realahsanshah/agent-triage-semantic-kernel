"""The one logging middleware function, implemented as an SK function-invocation filter.

SK's name for "middleware" is a Filter. It wraps every kernel-function call
(our billing/tech tools, and also the auto-generated Handoff-transfer_to_*
routing functions) with an await-next pattern: you run code before the call,
call `await next(context)` to actually invoke the function, then run code
after. Register it on a Kernel with:

    kernel.add_filter(FilterTypes.FUNCTION_INVOCATION, logging_filter)

This is the direct analog of a middleware function you'd pass into Agent
Framework's ChatAgent/workflow construction.
"""

from collections.abc import Awaitable, Callable

from semantic_kernel.filters import FunctionInvocationContext


async def logging_filter(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    plugin_name = context.function.plugin_name
    function_name = context.function.name
    args = dict(context.arguments)

    print(f"[middleware] -> calling {plugin_name}.{function_name}({args})")
    await next(context)
    print(f"[middleware] <- {plugin_name}.{function_name} returned: {context.result}")
