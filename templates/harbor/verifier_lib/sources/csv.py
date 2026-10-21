from pathlib import Path

from pydantic import field_validator

from ..models import StrictModel
from ..source_types import SourceCommand, SourceContext
from ._content_limits import cap_with_notice


class CsvExtractTextInput(StrictModel):
    """Input for csv.extract_text.

    Attributes:
        path: Workspace-relative CSV path.
    """

    path: str

    @field_validator("path")
    @classmethod
    def require_csv_extension(cls, value: str) -> str:
        """Requires the path to target a CSV file.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a CSV extension.

        Raises:
            ValueError: If the path does not end in ``.csv``.
        """
        if Path(value).suffix.lower() != ".csv":
            raise ValueError("csv.extract_text requires a .csv file")
        return value


class CsvExtractTextOutput(StrictModel):
    """Output from csv.extract_text.

    Attributes:
        text: Raw UTF-8 CSV text.
    """

    text: str


class ExtractText(SourceCommand):
    """Reads raw UTF-8 CSV files."""

    name = "extract_text"
    input_model = CsvExtractTextInput
    output_model = CsvExtractTextOutput

    def run(self, source_input: CsvExtractTextInput, context: SourceContext) -> CsvExtractTextOutput:
        """Reads CSV text from a workspace file.

        Args:
            source_input: Validated CSV extraction input.
            context: Source runtime context.

        Returns:
            Raw CSV text capped to the configured source content limit. If the file is longer
            than the cap, a truncation notice is appended so the cut is visible to the grader
            instead of silent.

        """
        resolved = context.resolve_path(source_input.path)
        raw = resolved.read_text(encoding="utf-8")
        return CsvExtractTextOutput(text=cap_with_notice(raw, context.max_content_chars))


COMMANDS = (ExtractText(),)
