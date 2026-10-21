from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, RootModel


SourceOutput = BaseModel | RootModel[Any]


class SourceAuthoringError(ValueError):
    """Raised when verifier-authored source configuration is invalid."""


@dataclass(frozen=True)
class SourceContext:
    """Runtime context shared by source commands.

    Attributes:
        workspace_dir: Root directory where task output artifacts live.
        agent_logs_dir: Root directory where Harbor exposes agent log artifacts.
        max_content_chars: Maximum text content a source should return.
    """

    workspace_dir: Path
    agent_logs_dir: Path
    max_content_chars: int

    def resolve_path(self, path: str) -> Path:
        """Resolves a workspace-relative path safely.

        Args:
            path: Workspace-relative path from a source command input.

        Returns:
            Absolute resolved path inside the workspace.

        Raises:
            SourceAuthoringError: If the path is absolute or escapes the
                workspace.
        """
        candidate = Path(path)
        if candidate.is_absolute():
            raise SourceAuthoringError("source path must be relative to workspace")

        workspace = self.workspace_dir.resolve()
        resolved = (workspace / candidate).resolve()
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise SourceAuthoringError("source path must stay inside workspace") from exc
        return resolved


class SourceCommand(ABC):
    """Base class for a command exposed by a source namespace."""

    name: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]
    output_model: ClassVar[type[BaseModel] | type[RootModel[Any]]]

    @abstractmethod
    def run(self, source_input: BaseModel, context: SourceContext) -> SourceOutput:
        """Runs the source command.

        Args:
            source_input: Validated command-specific input model.
            context: Runtime context with workspace and extraction limits.

        Returns:
            Validated command-specific output model.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class RegisteredSource:
    """Resolved source command registered under its public dotted name."""

    name: str
    command: SourceCommand

    @property
    def input_model(self) -> type[BaseModel]:
        """Returns the command input model class."""
        return self.command.input_model

    @property
    def output_model(self) -> type[BaseModel] | type[RootModel[Any]]:
        """Returns the command output model class."""
        return self.command.output_model
