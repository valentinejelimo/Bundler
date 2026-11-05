"""Unit tests for gdpval_qc.v18_spec — V18Spec.from_markdown parser.

Writes temp spec .md files (markdown + YAML frontmatter) and parses them.
Tests get_dimension, spec_hash, overlay handling (extends, remove_dimensions),
and malformed spec -> SpecValidationError.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gdpval_qc.v18_spec import (
    SpecValidationError, V18Spec, MIN_SPEC_VERSION_MAJOR,
    KNOWN_FRONTMATTER_KEYS,
)


# ── Spec text builders ────────────────────────────────────────────────────────

VALID_FRONTMATTER = """\
---
name: test_spec
version: "18.0"
models:
  - model: gpt-5
    reasoning_effort: high
    max_tokens: 8000
field_types:
  instruction: text
field_bindings:
  instruction:
    source: file
    path: instruction.md
severity_scale:
  0: none
  1: minor
  2: major
  3: critical
require_evidence_quote: true
confidence_demotion:
  low_at_sev1: demote_to_0
  low_at_sev2_or_higher: route_to_review
---

## Task Structure

Some prose here.

### dim_a

```yaml
dimension_name: Dimension A
check_kind: llm
inputs: [instruction]
gate:
  role: blocking
  reject_at_severity: 3
  on_disagreement: review
```

This is the rubric prose for dim_a.

### dim_b

```yaml
dimension_name: Dimension B
check_kind: llm
inputs: [instruction]
gate:
  role: advisory
  reject_at_severity: 3
```

Rubric prose for dim_b.
"""


def _write_spec(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def valid_spec_file(tmp_path: Path) -> Path:
    return _write_spec(tmp_path, "spec.md", VALID_FRONTMATTER)


# ── from_markdown: valid parse ─────────────────────────────────────────────────

class TestFromMarkdownValid:
    def test_parses_name_and_version(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.name == "test_spec"
        assert spec.version == "18.0"

    def test_parses_models(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert len(spec.models) == 1
        assert spec.models[0].model == "gpt-5"
        assert spec.models[0].reasoning_effort == "high"
        assert spec.models[0].max_tokens == 8000

    def test_parses_dimensions(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert len(spec.dimensions) == 2
        assert spec.dimensions[0].name == "dim_a"
        assert spec.dimensions[0].dimension_name == "Dimension A"
        assert spec.dimensions[0].gate.role == "blocking"
        assert spec.dimensions[0].gate.reject_at_severity == 3
        assert spec.dimensions[1].name == "dim_b"
        assert spec.dimensions[1].gate.role == "advisory"

    def test_dimension_prose_extracted(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert "rubric prose for dim_a" in spec.dimensions[0].prose

    def test_source_path_set(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.source_path == valid_spec_file.resolve()

    def test_field_types(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.field_types == {"instruction": "text"}

    def test_severity_scale(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.severity_scale[0] == "none"
        assert spec.severity_scale[3] == "critical"

    def test_require_evidence_quote(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.require_evidence_quote is True

    def test_confidence_demotion(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.confidence_demotion.low_at_sev1 == "demote_to_0"
        assert spec.confidence_demotion.low_at_sev2_or_higher == "route_to_review"


# ── get_dimension ─────────────────────────────────────────────────────────────

class TestGetDimension:
    def test_found(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        d = spec.get_dimension("dim_a")
        assert d is not None
        assert d.name == "dim_a"

    def test_not_found(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.get_dimension("nonexistent") is None


# ── spec_hash ────────────────────────────────────────────────────────────────

class TestSpecHash:
    def test_returns_hex_string(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        h = spec.spec_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_deterministic(self, valid_spec_file: Path):
        spec = V18Spec.from_markdown(valid_spec_file)
        assert spec.spec_hash() == spec.spec_hash()

    def test_different_specs_different_hash(self, tmp_path: Path):
        spec1 = V18Spec.from_markdown(valid_spec_file := _write_spec(
            tmp_path, "s1.md", VALID_FRONTMATTER))
        modified = VALID_FRONTMATTER.replace('name: test_spec', 'name: other_spec')
        spec2 = V18Spec.from_markdown(_write_spec(tmp_path, "s2.md", modified))
        assert spec1.spec_hash() != spec2.spec_hash()


# ── Malformed specs ────────────────────────────────────────────────────────────

class TestMalformedSpec:
    def test_missing_frontmatter(self, tmp_path: Path):
        p = _write_spec(tmp_path, "no_fm.md", "Just markdown, no frontmatter.")
        with pytest.raises(SpecValidationError):
            V18Spec.from_markdown(p)

    def test_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            V18Spec.from_markdown(tmp_path / "nonexistent.md")

    def test_old_version_rejected(self, tmp_path: Path):
        old = VALID_FRONTMATTER.replace('version: "18.0"', 'version: "17.0"')
        p = _write_spec(tmp_path, "old.md", old)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "18+" in str(exc_info.value) or "v17" in str(exc_info.value)

    def test_missing_version(self, tmp_path: Path):
        no_ver = VALID_FRONTMATTER.replace('version: "18.0"\n', '')
        p = _write_spec(tmp_path, "no_ver.md", no_ver)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "version" in str(exc_info.value).lower()

    def test_missing_name(self, tmp_path: Path):
        no_name = VALID_FRONTMATTER.replace('name: test_spec\n', '')
        p = _write_spec(tmp_path, "no_name.md", no_name)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "name" in str(exc_info.value).lower()

    def test_unknown_frontmatter_key(self, tmp_path: Path):
        with_unknown = VALID_FRONTMATTER.replace(
            'require_evidence_quote: true',
            'require_evidence_quote: true\nbogus_key: value')
        p = _write_spec(tmp_path, "unknown.md", with_unknown)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "bogus_key" in str(exc_info.value)

    def test_no_dimensions(self, tmp_path: Path):
        # Spec with no ### dim sections
        no_dims = """\
---
name: test_spec
version: "18.0"
models:
  - model: gpt-5
severity_scale:
  0: none
  1: minor
  2: major
  3: critical
---

No dimensions here.
"""
        p = _write_spec(tmp_path, "no_dims.md", no_dims)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "dimension" in str(exc_info.value).lower()

    def test_invalid_yaml_frontmatter(self, tmp_path: Path):
        bad_yaml = """\
---
name: test_spec
version: "18.0"
models:
  - model: gpt-5
  : bad yaml: [unclosed
---

### dim_a

```yaml
dimension_name: A
gate:
  role: blocking
  reject_at_severity: 3
```
"""
        p = _write_spec(tmp_path, "bad_yaml.md", bad_yaml)
        with pytest.raises(SpecValidationError):
            V18Spec.from_markdown(p)

    def test_missing_severity_level(self, tmp_path: Path):
        incomplete = VALID_FRONTMATTER.replace(
            "  3: critical\n", "")
        p = _write_spec(tmp_path, "incomplete.md", incomplete)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "severity" in str(exc_info.value).lower()

    def test_empty_models(self, tmp_path: Path):
        no_models = VALID_FRONTMATTER.replace(
            "  - model: gpt-5\n    reasoning_effort: high\n    max_tokens: 8000\n",
            "")
        p = _write_spec(tmp_path, "no_models.md", no_models)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "models" in str(exc_info.value).lower()

    def test_duplicate_dimension_names(self, tmp_path: Path):
        dup = VALID_FRONTMATTER + """
### dim_a

```yaml
dimension_name: Duplicate A
gate:
  role: blocking
  reject_at_severity: 3
```

Duplicate prose.
"""
        p = _write_spec(tmp_path, "dup.md", dup)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "duplicate" in str(exc_info.value).lower()

    def test_invalid_gate_role(self, tmp_path: Path):
        bad = VALID_FRONTMATTER.replace(
            "  role: blocking\n  reject_at_severity: 3",
            "  role: super_blocking\n  reject_at_severity: 3")
        p = _write_spec(tmp_path, "bad_role.md", bad)
        with pytest.raises(SpecValidationError):
            V18Spec.from_markdown(p)


# ── Overlay: extends ──────────────────────────────────────────────────────────

BASE_SPEC = """\
---
name: base_spec
version: "18.0"
models:
  - model: gpt-5
field_types:
  instruction: text
field_bindings:
  instruction:
    source: file
    path: instruction.md
severity_scale:
  0: none
  1: minor
  2: major
  3: critical
require_evidence_quote: true
confidence_demotion:
  low_at_sev1: demote_to_0
  low_at_sev2_or_higher: route_to_review
---

### dim_a

```yaml
dimension_name: Dimension A
gate:
  role: blocking
  reject_at_severity: 3
```

Base prose for dim_a.

### dim_b

```yaml
dimension_name: Dimension B
gate:
  role: advisory
  reject_at_severity: 3
```

Base prose for dim_b.
"""


class TestExtends:
    def test_overlay_inherits_base(self, tmp_path: Path):
        base = _write_spec(tmp_path, "base.md", BASE_SPEC)
        overlay = """\
---
extends: base.md
name: overlay_spec
version: "18.0"
---

### dim_c

```yaml
dimension_name: Dimension C
gate:
  role: blocking
  reject_at_severity: 3
```

Overlay-only prose for dim_c.
"""
        p = _write_spec(tmp_path, "overlay.md", overlay)
        spec = V18Spec.from_markdown(p)
        assert spec.name == "overlay_spec"
        dim_names = {d.name for d in spec.dimensions}
        assert "dim_a" in dim_names  # inherited
        assert "dim_b" in dim_names  # inherited
        assert "dim_c" in dim_names  # new

    def test_remove_dimensions(self, tmp_path: Path):
        _write_spec(tmp_path, "base.md", BASE_SPEC)
        overlay = """\
---
extends: base.md
name: overlay_spec
version: "18.0"
remove_dimensions: [dim_b]
---

### dim_c

```yaml
dimension_name: Dimension C
gate:
  role: blocking
  reject_at_severity: 3
```

Overlay prose.
"""
        p = _write_spec(tmp_path, "overlay.md", overlay)
        spec = V18Spec.from_markdown(p)
        dim_names = {d.name for d in spec.dimensions}
        assert "dim_a" in dim_names
        assert "dim_b" not in dim_names  # removed
        assert "dim_c" in dim_names

    def test_overlay_replaces_dim_section(self, tmp_path: Path):
        _write_spec(tmp_path, "base.md", BASE_SPEC)
        overlay = """\
---
extends: base.md
name: overlay_spec
version: "18.0"
---

### dim_a

```yaml
dimension_name: Dimension A Overridden
gate:
  role: advisory
  reject_at_severity: 3
```

Overridden prose.
"""
        p = _write_spec(tmp_path, "overlay.md", overlay)
        spec = V18Spec.from_markdown(p)
        dim_a = spec.get_dimension("dim_a")
        assert dim_a is not None
        assert dim_a.dimension_name == "Dimension A Overridden"
        assert dim_a.gate.role == "advisory"
        assert "Overridden prose" in dim_a.prose

    def test_circular_extends_detected(self, tmp_path: Path):
        overlay = """\
---
extends: self.md
name: overlay_spec
version: "18.0"
---

### dim_a

```yaml
dimension_name: A
gate:
  role: blocking
  reject_at_severity: 3
```

Prose.
"""
        p = _write_spec(tmp_path, "self.md", overlay)
        with pytest.raises(SpecValidationError) as exc_info:
            V18Spec.from_markdown(p)
        assert "circular" in str(exc_info.value).lower() or "Circular" in str(exc_info.value)

    def test_extends_not_found(self, tmp_path: Path):
        overlay = """\
---
extends: nonexistent.md
name: overlay_spec
version: "18.0"
---

### dim_a

```yaml
dimension_name: A
gate:
  role: blocking
  reject_at_severity: 3
```

Prose.
"""
        p = _write_spec(tmp_path, "overlay.md", overlay)
        with pytest.raises(SpecValidationError):
            V18Spec.from_markdown(p)


# ── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_min_version_is_18(self):
        assert MIN_SPEC_VERSION_MAJOR == 18

    def test_known_frontmatter_keys(self):
        assert "name" in KNOWN_FRONTMATTER_KEYS
        assert "version" in KNOWN_FRONTMATTER_KEYS
        assert "models" in KNOWN_FRONTMATTER_KEYS
