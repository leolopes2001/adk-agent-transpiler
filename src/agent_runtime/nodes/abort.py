from typing import Any

from google.adk import Context, Event
from google.adk.workflow import FunctionNode

from agent_runtime.nodes._common import function_node, parse_input
from agent_runtime.nodes.present import RESULT_STATE_KEY, failure_result, run_context
from agent_runtime.schemas import RunFailure


def abort(*, name: str) -> FunctionNode:
    async def run(ctx: Context, node_input: Any) -> Event:
        result = failure_result(parse_input(RunFailure, node_input), run_context(ctx))
        ctx.state[RESULT_STATE_KEY] = result.model_dump(mode="json")
        return Event(output=result)

    return function_node(name, run)
