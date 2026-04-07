from typing import Callable, Dict, Any


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}

    def register(self, name: str, handler: Callable[[Dict[str, Any]], Dict[str, Any]]):
        self._tools[name] = handler

    def invoke(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self._tools:
            return {"success": False, "error": f"unknown tool: {name}"}
        return self._tools[name](args)
