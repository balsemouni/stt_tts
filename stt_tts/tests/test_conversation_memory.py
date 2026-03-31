"""
test_conversation_memory.py — Unit tests for cag/conversation_memory.py
  • Message & UserProfile dataclasses
  • ConversationMemory: add_message, history, name extraction, clear, reset, format

Run:
    pytest tests/test_conversation_memory.py -v
"""

import sys
import os
import tempfile
import pytest
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cag"))

from conversation_memory import Message, UserProfile, ConversationMemory  # noqa


# ─── Minimal config stub ─────────────────────────────────────────────────────

@dataclass
class _FakeConfig:
    cache_file_path: str = ""
    enable_cache_persistence: bool = False
    verbose: bool = False


def _make_memory(max_history=10, tmpdir=None):
    cfg = _FakeConfig()
    if tmpdir:
        cfg.cache_file_path = os.path.join(tmpdir, "cache.pt")
    else:
        cfg.cache_file_path = os.path.join(tempfile.mkdtemp(), "cache.pt")
    return ConversationMemory(cfg, max_history=max_history)


# ═══════════════════════════════════════════════════════════════════════════════
#  Message Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessage:
    def test_create_message(self):
        m = Message(role="user", content="Hello")
        assert m.role == "user"
        assert m.content == "Hello"
        assert m.timestamp is not None

    def test_to_dict(self):
        m = Message(role="assistant", content="Hi there")
        d = m.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
        assert "timestamp" in d

    def test_metadata(self):
        m = Message(role="user", content="test", metadata={"latency": 100})
        assert m.metadata["latency"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
#  UserProfile Dataclass Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUserProfile:
    def test_default_values(self):
        p = UserProfile()
        assert p.name is None
        assert p.preferences == {}
        assert p.total_interactions == 0

    def test_to_dict(self):
        p = UserProfile(name="Alice")
        d = p.to_dict()
        assert d["name"] == "Alice"

    def test_from_dict(self):
        d = {"name": "Bob", "preferences": {"lang": "fr"},
             "first_interaction": "2025-01-01", "last_interaction": "2025-01-02",
             "total_interactions": 5}
        p = UserProfile.from_dict(d)
        assert p.name == "Bob"
        assert p.total_interactions == 5


# ═══════════════════════════════════════════════════════════════════════════════
#  ConversationMemory Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversationMemory:

    def test_init_empty(self):
        mem = _make_memory()
        assert len(mem.messages) == 0
        assert mem.user_profile.name is None

    def test_add_message(self):
        mem = _make_memory()
        mem.add_message("user", "Hello")
        assert len(mem.messages) == 1
        assert mem.messages[0].role == "user"
        assert mem.messages[0].content == "Hello"

    def test_add_message_increments_interactions(self):
        mem = _make_memory()
        mem.add_message("user", "Hello")
        assert mem.user_profile.total_interactions == 1

    def test_add_assistant_message_no_increment(self):
        mem = _make_memory()
        mem.add_message("assistant", "Hi!")
        assert mem.user_profile.total_interactions == 0

    def test_history_limit(self):
        """Messages should be trimmed to max_history * 2."""
        mem = _make_memory(max_history=3)
        for i in range(20):
            mem.add_message("user", f"msg {i}")
        assert len(mem.messages) <= 6  # max_history * 2

    def test_get_conversation_history_all(self):
        mem = _make_memory()
        mem.add_message("user", "a")
        mem.add_message("assistant", "b")
        history = mem.get_conversation_history()
        assert len(history) == 2

    def test_get_conversation_history_last_n(self):
        mem = _make_memory()
        for i in range(10):
            mem.add_message("user", f"msg{i}")
        history = mem.get_conversation_history(last_n=2)
        assert len(history) == 4  # last_n * 2

    def test_format_conversation_for_prompt_empty(self):
        mem = _make_memory()
        assert mem.format_conversation_for_prompt() == ""

    def test_format_conversation_for_prompt(self):
        mem = _make_memory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi there!")
        formatted = mem.format_conversation_for_prompt()
        assert "user" in formatted
        assert "assistant" in formatted
        assert "Hello" in formatted
        assert "Hi there!" in formatted

    def test_format_uses_llama_tokens(self):
        mem = _make_memory()
        mem.add_message("user", "test")
        formatted = mem.format_conversation_for_prompt()
        assert "<|start_header_id|>" in formatted
        assert "<|end_header_id|>" in formatted
        assert "<|eot_id|>" in formatted

    def test_get_stage_instruction_stub(self):
        mem = _make_memory()
        assert mem.get_stage_instruction() == ""

    # ── Name extraction ────────────────────────────────────────────────────────

    def test_extract_name_my_name_is(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("My name is Sarah") == "Sarah"

    def test_extract_name_im(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("I'm Alex") == "Alex"

    def test_extract_name_call_me(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("Call me Mike") == "Mike"

    def test_extract_name_just_call_me(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("just call me Tom") == "Tom"

    def test_extract_name_bare_name(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("David") == "David"

    def test_extract_name_returns_none_for_common_words(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("yes") is None
        assert mem.extract_name_from_response("hello") is None
        assert mem.extract_name_from_response("no") is None

    def test_extract_name_returns_none_for_sentence(self):
        mem = _make_memory()
        result = mem.extract_name_from_response("I want to buy a product for my business")
        assert result is None

    def test_extract_name_capitalizes(self):
        mem = _make_memory()
        assert mem.extract_name_from_response("my name is alice") == "Alice"

    def test_extract_name_people_usually_call_me(self):
        mem = _make_memory()
        result = mem.extract_name_from_response("people usually call me Sam")
        assert result == "Sam"

    # ── Set user name ──────────────────────────────────────────────────────────

    def test_set_user_name(self):
        mem = _make_memory()
        mem.set_user_name("Alice")
        assert mem.user_profile.name == "Alice"

    # ── Clear / Reset ──────────────────────────────────────────────────────────

    def test_clear_conversation(self):
        mem = _make_memory()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi")
        mem.set_user_name("Test")
        mem.clear_conversation()
        assert len(mem.messages) == 0
        assert mem.user_profile.name == "Test"  # profile kept

    def test_reset_all(self):
        mem = _make_memory()
        mem.add_message("user", "Hello")
        mem.set_user_name("Test")
        mem.reset_all()
        assert len(mem.messages) == 0
        assert mem.user_profile.name is None

    # ── Stats ──────────────────────────────────────────────────────────────────

    def test_get_stats(self):
        mem = _make_memory()
        mem.add_message("user", "Hello")
        mem.set_user_name("Alice")
        stats = mem.get_stats()
        assert stats["total_messages"] == 1
        assert stats["user_name"] == "Alice"
        assert stats["total_interactions"] == 1

    # ── Persistence ────────────────────────────────────────────────────────────

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        cfg = _FakeConfig(
            cache_file_path=os.path.join(tmpdir, "cache.pt"),
            enable_cache_persistence=True,
            verbose=False,
        )
        mem1 = ConversationMemory(cfg, max_history=10)
        mem1.add_message("user", "Hello")
        mem1.set_user_name("Alice")
        mem1.save_memory()

        # Load into new instance
        mem2 = ConversationMemory(cfg, max_history=10)
        assert len(mem2.messages) == 1
        assert mem2.messages[0].content == "Hello"
        assert mem2.user_profile.name == "Alice"
