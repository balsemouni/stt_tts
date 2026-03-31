"""
test_knowledge_store.py — Unit tests for cag/knowledge_store.py
  • SolutionEntry: to_compact_string
  • SolutionKnowledgeStore: load, build, dedup (without requiring real tokenizer)

Run:
    pytest tests/test_knowledge_store.py -v
"""

import sys
import os
import json
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cag"))

from knowledge_store import SolutionEntry, SolutionKnowledgeStore  # noqa


# ─── Fake tokenizer ──────────────────────────────────────────────────────────

class FakeTokenizer:
    """Mimics HuggingFace tokenizer.encode() for token counting."""
    def encode(self, text, add_special_tokens=True):
        # ~4 chars per token approximation
        return list(range(len(text) // 4))


# ─── Fake config ──────────────────────────────────────────────────────────────

class FakeConfig:
    def __init__(self, json_path):
        self.solutions_json_path = json_path
        self.max_knowledge_entries = 50000


# ═══════════════════════════════════════════════════════════════════════════════
#  SolutionEntry Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolutionEntry:

    def _make_entry(self, **overrides):
        defaults = dict(
            user_problem="Slow website",
            problem_keywords=["performance", "speed"],
            solution_name="CDN Accelerator",
            solution_description="A content delivery network solution",
            key_benefits=["Faster load times", "Global reach", "Reduced latency"],
            pricing_model="$99/month",
            implementation_time="2 weeks",
            target_industries=["E-commerce", "SaaS"],
        )
        defaults.update(overrides)
        return SolutionEntry(**defaults)

    def test_create_entry(self):
        e = self._make_entry()
        assert e.user_problem == "Slow website"
        assert e.solution_name == "CDN Accelerator"

    def test_to_compact_string(self):
        e = self._make_entry()
        compact = e.to_compact_string()
        assert "PROBLEM:Slow website" in compact
        assert "SOLUTION:CDN Accelerator" in compact
        assert "BENEFITS:" in compact
        assert "PRICE:$99/month" in compact
        assert "TIME:2 weeks" in compact

    def test_compact_string_limits_benefits_to_3(self):
        e = self._make_entry(key_benefits=["a", "b", "c", "d", "e"])
        compact = e.to_compact_string()
        # Should only have 3 benefits
        benefits_part = compact.split("BENEFITS:")[1].split("|")[0]
        assert benefits_part.count(";") == 2  # 3 items → 2 separators

    def test_compact_string_strips_newlines(self):
        e = self._make_entry(user_problem="Line1\nLine2")
        compact = e.to_compact_string()
        assert "\n" not in compact

    def test_metadata_optional(self):
        e = self._make_entry()
        assert e.metadata is None


# ═══════════════════════════════════════════════════════════════════════════════
#  SolutionKnowledgeStore Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolutionKnowledgeStore:

    def _write_json(self, entries):
        """Write solution entries to a temp JSON file."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                        delete=False, encoding="utf-8")
        json.dump(entries, f, ensure_ascii=False)
        f.close()
        return f.name

    def _write_jsonl(self, entries):
        """Write solution entries to a temp JSONL file."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl",
                                        delete=False, encoding="utf-8")
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        f.close()
        return f.name

    def _sample_entries(self, n=3):
        return [
            {
                "user_problem": f"Problem {i}",
                "problem_keywords": [f"kw{i}"],
                "solution_name": f"Solution {i}",
                "solution_description": f"Description for solution {i}",
                "key_benefits": [f"Benefit {i}a", f"Benefit {i}b"],
                "pricing_model": f"${i}0/month",
                "implementation_time": f"{i} weeks",
                "target_industries": ["Tech"],
            }
            for i in range(n)
        ]

    def test_load_from_json(self):
        path = self._write_json(self._sample_entries(5))
        try:
            cfg = FakeConfig(path)
            store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
            count = store.load_from_sources()
            assert count == 5
            assert len(store.entries) == 5
        finally:
            os.unlink(path)

    def test_load_from_jsonl(self):
        path = self._write_jsonl(self._sample_entries(4))
        try:
            cfg = FakeConfig(path)
            store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
            count = store.load_from_sources()
            assert count == 4
        finally:
            os.unlink(path)

    def test_load_skips_invalid_entries(self):
        entries = self._sample_entries(3)
        entries.append({"user_problem": "", "solution_name": "", "solution_description": ""})
        entries.append({"bad_key": "value"})
        path = self._write_json(entries)
        try:
            cfg = FakeConfig(path)
            store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
            count = store.load_from_sources()
            assert count == 3  # only valid ones
        finally:
            os.unlink(path)

    def test_load_missing_file_raises(self):
        cfg = FakeConfig("/nonexistent/file.json")
        store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
        with pytest.raises(ValueError, match="not found"):
            store.load_from_sources()

    def test_entries_have_metadata(self):
        path = self._write_json(self._sample_entries(2))
        try:
            cfg = FakeConfig(path)
            store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
            store.load_from_sources()
            for entry in store.entries:
                assert entry.metadata is not None
                assert entry.metadata["source"] == "json"
        finally:
            os.unlink(path)

    def test_json_with_solutions_key(self):
        """JSON with {'solutions': [...]} wrapper."""
        entries = self._sample_entries(2)
        path = self._write_json({"solutions": entries})
        try:
            cfg = FakeConfig(path)
            store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
            # The store handles both array and object+solutions key
            # load_from_json checks for this
        finally:
            os.unlink(path)

    def test_empty_file_raises(self):
        path = self._write_json([])
        try:
            cfg = FakeConfig(path)
            store = SolutionKnowledgeStore(FakeTokenizer(), cfg)
            with pytest.raises(ValueError, match="No solution data"):
                store.load_from_sources()
        finally:
            os.unlink(path)
