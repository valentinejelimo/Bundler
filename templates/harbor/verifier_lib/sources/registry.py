from pathlib import Path
from types import ModuleType

from pydantic import ValidationError

from ..models import Source
from ..source_types import RegisteredSource, SourceAuthoringError, SourceCommand, SourceContext, SourceOutput
from . import csv, docx, filesystem, json, md, pdf, pptx, response, text, xlsx


SOURCE_NAMESPACE_MODULES = (filesystem, json, docx, xlsx, pdf, pptx, text, md, csv, response)


class SourceRegistry:
    """Registry that resolves verifier source calls to command objects.

    Source commands live in namespace modules such as ``sources/json.py`` and
    ``sources/document.py``. The registry is the small lookup layer between
    the author-facing verifier.json DSL and those command implementations: it
    converts a dotted source name like ``json.read_file`` into the matching
    command class, validates the command arguments with that command's input
    model, executes it with the shared source context, and validates the output.

    The runner uses this class for every verifier source. Source namespace
    files still own the actual behavior; this class only centralizes discovery,
    validation, and consistent error reporting for unknown source names or
    invalid arguments.
    """

    def __init__(self, workspace_dir: Path, agent_logs_dir: Path | None = None, max_content_chars: int = 90000):
        """Initializes the source registry.

        Args:
            workspace_dir: Workspace containing task output artifacts.
            agent_logs_dir: Harbor agent log artifact directory. Response
                sources read the final agent message from this directory.
            max_content_chars: Maximum text content returned by document
                sources.

        Raises:
            ValueError: If two source commands resolve to the same dotted name.
        """
        self.context = SourceContext(
            workspace_dir=workspace_dir,
            agent_logs_dir=agent_logs_dir or Path("/logs/agent"),
            max_content_chars=max_content_chars,
        )
        self._sources: dict[str, RegisteredSource] = {}
        for module in SOURCE_NAMESPACE_MODULES:
            namespace = namespace_from_module(module)
            for command in commands_from_module(module):
                full_name = f"{namespace}.{command.name}"
                if full_name in self._sources:
                    raise ValueError(f"duplicate source definition: {full_name}")
                self._sources[full_name] = RegisteredSource(
                    name=full_name,
                    command=command,
                )
    def get_definition(self, name: str) -> RegisteredSource:
        """Returns the registered source definition for a dotted name.

        Args:
            name: Dotted source command name.

        Returns:
            Registered source definition.

        Raises:
            SourceAuthoringError: If no source exists with that name.
        """
        definition = self._sources.get(name)
        if definition is None:
            raise SourceAuthoringError(f"unknown source: {name}")
        return definition

    def run(self, source: Source) -> SourceOutput:
        """Validates arguments and executes a source command.

        Args:
            source: Declarative source call from verifier.json.

        Returns:
            Validated source command output.

        Raises:
            SourceAuthoringError: If the source name is unknown or arguments are
                invalid.
        """
        source_name = source.source_name
        definition = self.get_definition(source_name)
        try:
            source_input = definition.input_model.model_validate(source.arguments)
        except ValidationError as exc:
            raise SourceAuthoringError(f"source argument validation failed for {source_name}: {exc}") from exc

        output = definition.command.run(source_input, self.context)
        return definition.output_model.model_validate(output)


def namespace_from_module(module: ModuleType) -> str:
    """Derives a source namespace from a module name.

    Args:
        module: Source namespace module.

    Returns:
        Final module path segment.
    """
    return module.__name__.rsplit(".", 1)[-1]


def commands_from_module(module: ModuleType) -> tuple[SourceCommand, ...]:
    """Reads source command instances from a namespace module.

    Args:
        module: Source namespace module.

    Returns:
        Tuple of source command instances.

    Raises:
        ValueError: If the module does not define ``COMMANDS``.
    """
    commands = getattr(module, "COMMANDS", None)
    if commands is None:
        raise ValueError(f"source namespace {namespace_from_module(module)} does not define COMMANDS")
    return tuple(commands)
