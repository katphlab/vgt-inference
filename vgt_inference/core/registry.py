from __future__ import annotations

from collections.abc import Callable


class Registry:
    def __init__(self, name: str) -> None:
        self._name = name
        self._obj_map: dict[str, Callable] = {}

    def register(self, name: str | None = None):
        def decorator(obj: Callable) -> Callable:
            key = name or obj.__name__
            if key in self._obj_map:
                raise KeyError(f"An object named '{key}' is already registered in '{self._name}'")
            self._obj_map[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable:
        try:
            return self._obj_map[name]
        except KeyError as exc:
            raise KeyError(f"No object named '{name}' found in registry '{self._name}'") from exc
