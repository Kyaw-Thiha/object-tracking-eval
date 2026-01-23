"""Base backend contract for rendering RenderSpec objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schema.render_spec import RenderSpec


class BaseBackend(ABC):
    """Base class for backends that render RenderSpec objects."""

    @abstractmethod
    def render(self, spec: RenderSpec) -> Any: ...

    def update(self, handle: Any, spec: RenderSpec) -> None:
        """Optional update hook; default is a no-op."""
        _ = handle
        _ = spec
