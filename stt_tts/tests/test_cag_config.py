"""
test_cag_config.py — Unit tests for cag/cag_config.py
  • CAGConfig defaults, presets, validation

Run:
    pytest tests/test_cag_config.py -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cag"))

from cag_config import CAGConfig, COMPRESSED_SYSTEM_PROMPT  # noqa


class TestCompressedSystemPrompt:
    def test_prompt_is_string(self):
        assert isinstance(COMPRESSED_SYSTEM_PROMPT, str)

    def test_prompt_mentions_nova(self):
        assert "Nova" in COMPRESSED_SYSTEM_PROMPT

    def test_prompt_mentions_voice(self):
        assert "voice" in COMPRESSED_SYSTEM_PROMPT.lower()

    def test_prompt_is_reasonably_short(self):
        # Should be compressed — rough estimate: ~200 tokens ≈ ~800 chars
        assert len(COMPRESSED_SYSTEM_PROMPT) < 1500


class TestCAGConfig:

    def test_default_model(self):
        cfg = CAGConfig()
        assert "Llama-3.2-3B" in cfg.model_id

    def test_default_quantization(self):
        cfg = CAGConfig()
        assert cfg.use_4bit is True
        assert cfg.quant_type == "nf4"
        assert cfg.compute_dtype == "float16"

    def test_default_token_limits(self):
        cfg = CAGConfig()
        assert cfg.max_context_tokens == 4096
        assert cfg.max_new_tokens == 80
        assert cfg.model_max_tokens == 7000

    def test_default_generation_greedy(self):
        cfg = CAGConfig()
        assert cfg.temperature is None
        assert cfg.top_p is None
        assert cfg.top_k is None
        assert cfg.num_beams == 1
        assert cfg.repetition_penalty == 1.0

    def test_default_gpu_settings(self):
        cfg = CAGConfig()
        assert cfg.gpu_memory_fraction == 0.80
        assert cfg.min_free_memory_mb == 100

    def test_flash_attention_enabled_by_default(self):
        cfg = CAGConfig()
        assert cfg.use_flash_attention is True

    def test_tf32_enabled(self):
        cfg = CAGConfig()
        assert cfg.enable_tf32 is True

    def test_cache_persistence_enabled(self):
        cfg = CAGConfig()
        assert cfg.enable_cache_persistence is True

    def test_streaming_buffer_size_one(self):
        cfg = CAGConfig()
        assert cfg.streaming_buffer_size == 1

    def test_cache_overflow_policy(self):
        cfg = CAGConfig()
        assert cfg.cache_overflow_policy == "truncate"

    def test_knowledge_max_entries(self):
        cfg = CAGConfig()
        assert cfg.max_knowledge_entries == 50_000

    def test_custom_values(self):
        cfg = CAGConfig(
            max_new_tokens=256,
            max_context_tokens=2048,
            use_4bit=False,
        )
        assert cfg.max_new_tokens == 256
        assert cfg.max_context_tokens == 2048
        assert cfg.use_4bit is False
