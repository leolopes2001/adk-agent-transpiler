"""Import probe for a generated package, meant to run as `python -I -m transpiler.validation.probe`.

It imports the package with networking disabled, inspects the ADK objects it
built and prints one JSON document. Running it in a subprocess keeps the
transpiler itself free of google.adk imports and of whatever the generated
module does at import time; ADK modules are loaded dynamically for the same
reason (the import contracts forbid static imports of google.*).
"""

import importlib
import json
import socket
import sys
from pathlib import Path
from typing import Any

START_NAME = "__START__"

_network_attempts: list[str] = []


def _refuse(*args: Any, **kwargs: Any) -> Any:
    _network_attempts.append(repr(args[1:2] or args[:1]))
    raise OSError("network access is blocked during the import probe")


def block_network() -> None:
    socket.socket.connect = _refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = _refuse  # type: ignore[method-assign]
    socket.create_connection = _refuse
    socket.getaddrinfo = _refuse


def _qualified(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _edges(workflow: Any) -> list[list[str | None]]:
    def name(node: Any) -> str:
        return "START" if node.name == START_NAME else str(node.name)

    return sorted(
        ([name(e.from_node), name(e.to_node), e.route] for e in workflow.graph.edges),
        key=lambda edge: tuple(part or "" for part in edge),
    )


def inspect_package(package_dir: Path) -> dict[str, Any]:
    adk = importlib.import_module("google.adk")
    workflow_module = importlib.import_module("google.adk.workflow")
    apps = importlib.import_module("google.adk.apps")

    sys.path.insert(0, str(package_dir.parent))
    package = importlib.import_module(package_dir.name)
    agent = importlib.import_module(f"{package_dir.name}.agent")

    root = agent.root_agent
    app = agent.app
    members = {
        var: value
        for var, value in vars(agent).items()
        if isinstance(value, workflow_module.BaseNode) and var != "root_agent"
    }
    workflows = [root, *(v for v in members.values() if isinstance(v, adk.Workflow))]
    resumability = getattr(app, "resumability_config", None)
    return {
        "adk_version": adk.__version__,
        "root_agent_type": _qualified(root),
        "root_agent_is_workflow": isinstance(root, adk.Workflow),
        "root_agent_name": root.name,
        "app_type": _qualified(app),
        "app_is_app": isinstance(app, apps.App),
        "app_resumable": bool(resumability and resumability.is_resumable),
        "exports": sorted(getattr(package, "__all__", [])),
        "node_names": sorted(value.name for value in members.values()),
        "variables_match_names": all(var == value.name for var, value in members.items()),
        "workflows": {w.name: _edges(w) for w in workflows},
        "network_attempts": list(_network_attempts),
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        sys.stderr.write("usage: python -I -m transpiler.validation.probe <package-dir>\n")
        return 2
    sys.dont_write_bytecode = True
    block_network()
    result = inspect_package(Path(args[0]).resolve())
    sys.stdout.write(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
