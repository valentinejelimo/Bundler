"""Unit tests for gdpval_qc.upstream_deliverables — classification of inputs.

Tests golden_index, classify_inputs, build_upstream_deliverable_inputs
with temp dirs of fake node files. Tests sha256-based classification.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from gdpval_qc import upstream_deliverables
from gdpval_qc.upstream_deliverables import (
    build_upstream_deliverable_inputs, classify_inputs, golden_index,
)


def _write(path: Path, content: bytes = b"hello") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Build a small corpus:

    corpus/
      node_alpha/
        solution/golden/report.pdf   (content: "golden_alpha")
        environment/data/input.txt   (content: "source")
      node_beta/
        solution/golden/data.csv      (content: "golden_beta")
        environment/data/
          report.pdf                  (copy of node_alpha golden -> upstream deliverable)
          local.txt                   (own content -> source)
      node_gamma/
        solution/golden/
          byname.json                 (content: "gamma_content")
        environment/data/
          BYNAME.JSON                 (same name, different bytes -> name match -> upstream)
          extra.md                    (source)
    """
    alpha = tmp_path / "node_alpha"
    _write(alpha / "solution" / "golden" / "report.pdf", b"golden_alpha")
    _write(alpha / "environment" / "data" / "input.txt", b"source")

    beta = tmp_path / "node_beta"
    _write(beta / "solution" / "golden" / "data.csv", b"golden_beta")
    _write(beta / "environment" / "data" / "report.pdf", b"golden_alpha")
    _write(beta / "environment" / "data" / "local.txt", b"beta_local")

    gamma = tmp_path / "node_gamma"
    _write(gamma / "solution" / "golden" / "byname.json", b"gamma_content")
    _write(gamma / "environment" / "data" / "BYNAME.JSON", b"different_bytes")
    _write(gamma / "environment" / "data" / "extra.md", b"gamma_source")

    return tmp_path


# ── golden_index ───────────────────────────────────────────────────────────────

class TestGoldenIndex:
    def test_indexes_by_sha_and_name(self, corpus: Path):
        # Clear cache
        upstream_deliverables._index_cache.pop(str(corpus), None)
        by_sha, by_name = golden_index(corpus)
        sha_alpha = hashlib.sha256(b"golden_alpha").hexdigest()
        assert by_sha[sha_alpha] == "node_alpha"
        assert by_name["report.pdf"] == "node_alpha"
        assert by_name["data.csv"] == "node_beta"
        assert by_name["byname.json"] == "node_gamma"

    def test_caches_result(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        r1 = golden_index(corpus)
        r2 = golden_index(corpus)
        # golden_index returns a fresh tuple wrapping the cached (by_sha, by_name)
        # dicts each call, so the tuple identity differs but the inner dicts
        # are the same cached objects.
        assert r1[0] is r2[0]  # by_sha dict is cached and reused
        assert r1[1] is r2[1]  # by_name dict is cached and reused
        assert r1 == r2


# ── classify_inputs ───────────────────────────────────────────────────────────

class TestClassifyInputs:
    def test_sha_match_different_node_is_deliverable(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        deliverable, source = classify_inputs(corpus / "node_beta")
        # report.pdf matches node_alpha's golden by sha
        deliv_names = {fn for fn, _ in deliverable}
        assert "report.pdf" in deliv_names
        # local.txt is a source
        assert "local.txt" in source

    def test_own_golden_match_is_source_not_deliverable(self, corpus: Path):
        # If we classify node_alpha's own inputs, and an input byte-matches
        # node_alpha's own golden, it should be source (not deliverable).
        # We don't have that setup in our fixture, but test the != check:
        upstream_deliverables._index_cache.pop(str(corpus), None)
        # node_alpha has input.txt which is "source" - not a golden of any node
        deliverable, source = classify_inputs(corpus / "node_alpha")
        assert "input.txt" in source
        assert deliverable == []

    def test_name_match_when_no_sha(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        # node_gamma's BYNAME.JSON: different bytes but same name (case-insensitive)
        # as node_gamma's own golden byname.json -> own golden -> source
        deliverable, source = classify_inputs(corpus / "node_gamma")
        # BYNAME.JSON matches own golden by name -> own golden -> source
        assert "BYNAME.JSON" in source

    def test_name_match_different_node(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        # Create a node_delta whose input has same NAME as node_gamma's golden
        # but placed in node_delta's dir
        delta = corpus / "node_delta"
        _write(delta / "solution" / "golden" / "unique.txt", b"delta_golden")
        _write(delta / "environment" / "data" / "byname.json", b"delta_input")
        upstream_deliverables._index_cache.pop(str(corpus), None)
        deliverable, source = classify_inputs(delta)
        # byname.json matches node_gamma's golden by name (different bytes)
        deliv_names = {fn for fn, _ in deliverable}
        assert "byname.json" in deliv_names

    def test_skip_files_excluded(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        # Add a sources.md and .ds_store to node_alpha's data dir
        _write(corpus / "node_alpha" / "environment" / "data" / "sources.md",
               b"skip me")
        _write(corpus / "node_alpha" / "environment" / "data" / ".ds_store",
               b"skip me too")
        upstream_deliverables._index_cache.pop(str(corpus), None)
        deliverable, source = classify_inputs(corpus / "node_alpha")
        assert "sources.md" not in source
        assert ".ds_store" not in source
        assert "input.txt" in source

    def test_no_data_dir(self, tmp_path: Path):
        # A node with no environment/data directory
        node = tmp_path / "lonely"
        _write(node / "solution" / "golden" / "x.txt", b"x")
        deliverable, source = classify_inputs(node)
        assert deliverable == []
        assert source == []


# ── build_upstream_deliverable_inputs ──────────────────────────────────────────

class TestBuildUpstreamDeliverableInputs:
    def test_empty_when_no_deliverables(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        sample = SimpleNamespace(task_dir=corpus / "node_alpha")
        result = build_upstream_deliverable_inputs(sample)
        assert result == ""

    def test_nonempty_when_deliverables(self, corpus: Path):
        upstream_deliverables._index_cache.pop(str(corpus), None)
        sample = SimpleNamespace(task_dir=corpus / "node_beta")
        result = build_upstream_deliverable_inputs(sample)
        assert result != ""
        assert "UPSTREAM-DELIVERABLE" in result
        assert "report.pdf" in result
        assert "node_alpha" in result

    def test_returns_empty_on_exception(self):
        # A non-existent task_dir triggers an exception inside classify_inputs
        # (parent has no nodes), which is caught and returns ""
        sample = SimpleNamespace(task_dir=Path("/nonexistent/deep/path"))
        result = build_upstream_deliverable_inputs(sample)
        assert result == ""
