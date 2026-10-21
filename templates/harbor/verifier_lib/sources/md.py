from pathlib import Path

from pydantic import field_validator

from ..models import StrictModel
from ..source_types import SourceCommand, SourceContext
from ._content_limits import cap_with_notice


class MdExtractTextInput(StrictModel):
    """Input for md.extract_text.

    Attributes:
        path: Workspace-relative Markdown path.
    """

    path: str

    @field_validator("path")
    @classmethod
    def require_markdown_extension(cls, value: str) -> str:
        """Requires the path to target a Markdown file.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a Markdown extension.

        Raises:
            ValueError: If the path does not end in ``.md``.
        """
        if Path(value).suffix.lower() != ".md":
            raise ValueError("md.extract_text requires a .md file")
        return value


class MdExtractTextOutput(StrictModel):
    """Output from md.extract_text.

    Attributes:
        text: Raw UTF-8 Markdown text.
    """

    text: str


class ExtractText(SourceCommand):
    """Reads raw UTF-8 Markdown files."""

    name = "extract_text"
    input_model = MdExtractTextInput
    output_model = MdExtractTextOutput

    def run(self, source_input: MdExtractTextInput, context: SourceContext) -> MdExtractTextOutput:
        """Reads Markdown text from a workspace file.

        Args:
            source_input: Validated Markdown extraction input.
            context: Source runtime context.

        Returns:
            Raw Markdown text capped to the configured source content limit. If the file is
            longer than the cap, a truncation notice is appended so the cut is visible to the
            grader instead of silent.

        """
        resolved = context.resolve_path(source_input.path)
        raw = resolved.read_text(encoding="utf-8")
        return MdExtractTextOutput(text=cap_with_notice(raw, context.max_content_chars))


COMMANDS = (ExtractText(),)
