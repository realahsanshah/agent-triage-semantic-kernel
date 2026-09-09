"""SK Plugins: classes whose methods are exposed to the LLM as callable tools.

Each method below is decorated with @kernel_function. Semantic Kernel reads the
decorator's description plus the Annotated[...] parameter hints to build the
function-calling schema the model sees - conceptually the same job Agent
Framework's plain `tools=[fn1, fn2]` list does, just grouped into a class here.

All data is in-memory and fake; this is a demo, not a real billing/ops backend.
"""

from typing import Annotated

from semantic_kernel.functions import kernel_function

# customer_id -> balance in USD cents
_FAKE_BALANCES: dict[str, int] = {
    "cust_1": 4999,
    "cust_2": 12000,
}

# service_name -> status
_FAKE_SERVICE_STATUS: dict[str, str] = {
    "login": "degraded",
    "billing-api": "operational",
    "email": "operational",
}


class BillingPlugin:
    """Tools for the billing specialist agent."""

    @kernel_function(description="Check a customer's current account balance in USD.")
    def check_balance(
        self,
        customer_id: Annotated[str, "The customer's account ID, e.g. 'cust_1'."],
    ) -> str:
        cents = _FAKE_BALANCES.get(customer_id)
        if cents is None:
            return f"No account found for customer_id '{customer_id}'."
        return f"Customer {customer_id} has a balance of ${cents / 100:.2f}."

    @kernel_function(description="Issue a refund (full or partial) to a customer.")
    def issue_refund(
        self,
        customer_id: Annotated[str, "The customer's account ID, e.g. 'cust_1'."],
        amount_usd: Annotated[float, "Dollar amount to refund, e.g. 9.99."],
        reason: Annotated[str, "Short reason for the refund."],
    ) -> str:
        return (
            f"Refund of ${amount_usd:.2f} issued to customer {customer_id}. "
            f"Reason: {reason}."
        )


class TechSupportPlugin:
    """Tools for the technical-support specialist agent."""

    @kernel_function(description="Check whether a named service is currently up.")
    def check_system_status(
        self,
        service_name: Annotated[str, "Service name, e.g. 'login', 'billing-api', 'email'."],
    ) -> str:
        status = _FAKE_SERVICE_STATUS.get(service_name, "unknown")
        return f"Service '{service_name}' status: {status}."

    @kernel_function(description="Restart a named service.")
    def restart_service(
        self,
        service_name: Annotated[str, "Service name to restart, e.g. 'login'."],
    ) -> str:
        _FAKE_SERVICE_STATUS[service_name] = "operational"
        return f"Service '{service_name}' has been restarted and is now operational."
