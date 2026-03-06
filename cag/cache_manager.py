"""
CAG Architecture - Cache Manager Module for Solution Recommendations
Manages KV cache lifecycle with truncation and persistence

IMPROVEMENTS v2:
- truncate_to_knowledge(): torch.cuda.synchronize() and gc.collect() removed.
  This runs after EVERY query — calling synchronize() here blocked the GPU
  pipeline for 10-30 ms between responses, causing audible stutter in TTS.
  GPU memory is NOT freed by slicing tensors anyway, so the calls were pure overhead.
- _cleanup_memory() is unchanged (still used at startup/shutdown where latency
  doesn't matter).
- Handles both legacy tuple PKV and newer DynamicCache objects.
"""

import torch
import gc
import os
import json
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict


@dataclass
class CacheState:
    """Represents the state of KV cache"""
    input_ids: torch.Tensor
    token_count: int
    knowledge_token_count: int          # N tokens we preserve across queries
    past_key_values: Optional[Any] = None
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_count":           self.token_count,
            "knowledge_token_count": self.knowledge_token_count,
            "timestamp":             self.timestamp,
            "metadata":              self.metadata or {},
        }


class CacheManager:
    """
    Cache Manager — Core of CAG Architecture

    Responsibilities:
    1. Pre-compute KV cache from solution knowledge base (once at startup)
    2. Truncate cache after each query (reset to knowledge_token_count)
    3. Persist and load cache from disk
    4. Handle cache overflow according to policy
    """

    def __init__(self, model, tokenizer, device, config):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device
        self.config    = config

        self.cache_state: Optional[CacheState] = None
        self.is_initialized = False

    # ──────────────────────────────────────────────────────────────────────────
    # Pre-compute
    # ──────────────────────────────────────────────────────────────────────────

    def precompute_cache(self, knowledge_text: str) -> CacheState:
        """
        Pre-compute KV cache from solution knowledge base.

        Args:
            knowledge_text: Formatted solution knowledge (PROBLEM:|SOLUTION: format)
        """
        print("\n" + "=" * 60)
        print("🎯 PRECOMPUTING KV CACHE")
        print("=" * 60)

        self._cleanup_memory()
        free_before = torch.cuda.mem_get_info()[0] // 1024 ** 2
        print(f"📊 Free memory before cache: {free_before} MB")

        prompt = self._build_cache_prompt(knowledge_text)
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_context_tokens,
        ).input_ids.to(self.device)

        token_count = input_ids.shape[-1]
        print(f"📝 Knowledge base tokens: {token_count}")

        if token_count > self.config.max_context_tokens:
            print(f"⚠️  Truncating to {self.config.max_context_tokens} tokens")
            input_ids   = input_ids[:, :self.config.max_context_tokens]
            token_count = input_ids.shape[-1]

        try:
            with torch.no_grad():
                with torch.amp.autocast("cuda"):
                    outputs = self.model(input_ids, use_cache=True, return_dict=True)

            past_key_values = outputs.past_key_values
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()

            cache_state = CacheState(
                input_ids             = input_ids,
                token_count           = token_count,
                knowledge_token_count = token_count,
                past_key_values       = past_key_values,
                timestamp             = None,
                metadata              = {"source": "solution_knowledge_base", "type": "recommendations"},
            )

            del outputs
            self._cleanup_memory()

            free_after   = torch.cuda.mem_get_info()[0] // 1024 ** 2
            memory_used  = free_before - free_after
            print(f"✅ KV Cache pre-computed  |  {token_count} tokens  |  ~{memory_used} MB used")

            self.cache_state   = cache_state
            self.is_initialized = True
            return cache_state

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n❌ OUT OF MEMORY with {token_count} tokens!")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Truncate (hot path — called after every query)
    # ──────────────────────────────────────────────────────────────────────────

    def truncate_to_knowledge(self):
        """
        Truncate KV cache back to knowledge-base length (N tokens).

        Called after every query to strip the query + response tokens so the
        next query starts from a clean knowledge context.

        PERFORMANCE NOTE:
        torch.cuda.synchronize() and gc.collect() have been intentionally
        removed from this method.  They were called here in an earlier version
        and caused 10-30 ms of dead time between every streamed response
        (audible stutter in TTS pipelines).  GPU memory is NOT actually freed
        by tensor slicing — those calls gave zero benefit at significant cost.
        Heavy cleanup is deferred to reset / shutdown phases.
        """
        if not self.is_initialized or self.cache_state is None:
            raise ValueError("Cache not initialized.")

        N = self.cache_state.knowledge_token_count

        # Truncate input_ids
        if self.cache_state.input_ids.shape[-1] > N:
            self.cache_state.input_ids = self.cache_state.input_ids[:, :N]

        # Truncate past_key_values
        if self.cache_state.past_key_values is not None:
            new_pkv = []
            for layer_pair in self.cache_state.past_key_values:
                truncated = []
                for item in layer_pair:
                    if isinstance(item, torch.Tensor) and item.dim() >= 3:
                        # Standard layout: [batch, heads, seq_len, head_dim]
                        truncated.append(item[:, :, :N, :])
                    else:
                        truncated.append(item)
                new_pkv.append(tuple(truncated))
            self.cache_state.past_key_values = tuple(new_pkv)

        self.cache_state.token_count = N

    # ──────────────────────────────────────────────────────────────────────────
    # Overflow
    # ──────────────────────────────────────────────────────────────────────────

    def handle_overflow(self, query_tokens: int) -> bool:
        if self.cache_state is None:
            return False
        total     = self.cache_state.knowledge_token_count + query_tokens
        available = self.config.max_context_tokens + self.config.max_new_tokens
        if total <= available:
            return True
        if self.config.cache_overflow_policy == "error":
            raise ValueError(f"Query overflow: {total} > {available}")
        elif self.config.cache_overflow_policy == "truncate":
            return True
        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save_cache(self, path: Optional[str] = None):
        if self.cache_state is None:
            raise ValueError("No cache to save")

        path = path or self.config.cache_file_path

        pkv_cpu = None
        if self.cache_state.past_key_values is not None:
            pkv_cpu = tuple(
                tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in layer)
                for layer in self.cache_state.past_key_values
            )

        torch.save(
            {
                "input_ids":        self.cache_state.input_ids.cpu(),
                "past_key_values":  pkv_cpu,
                "metadata":         self.cache_state.to_dict(),
            },
            path,
        )

        if self.config.verbose:
            print(f"💾 Cache saved → {path}")

    def load_cache(self, path: Optional[str] = None) -> bool:
        path = path or self.config.cache_file_path

        if not os.path.exists(path):
            return False

        try:
            cache_data = torch.load(path, map_location=self.device, weights_only=False)
            metadata   = cache_data["metadata"]

            pkv_raw = cache_data.get("past_key_values")
            if pkv_raw is None:
                print("⚠️  Cache is stale (no past_key_values) — rebuilding...")
                return False

            past_key_values = tuple(
                tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in layer)
                for layer in pkv_raw
            )

            self.cache_state = CacheState(
                input_ids             = cache_data["input_ids"].to(self.device),
                token_count           = metadata["token_count"],
                knowledge_token_count = metadata["knowledge_token_count"],
                past_key_values       = past_key_values,
                timestamp             = metadata.get("timestamp"),
                metadata              = metadata.get("metadata"),
            )
            self.is_initialized = True

            if self.config.verbose:
                print(f"✅ Cache loaded ← {path}")
            return True

        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Info
    # ──────────────────────────────────────────────────────────────────────────

    def get_cache_info(self) -> Dict[str, Any]:
        if self.cache_state is None:
            return {"initialized": False}
        return {
            "initialized":     self.is_initialized,
            "token_count":     self.cache_state.token_count,
            "knowledge_tokens": self.cache_state.knowledge_token_count,
            "metadata":        self.cache_state.metadata,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_cache_prompt(self, knowledge_text: str) -> str:
        """
        Build the prompt that gets baked into the KV cache.

        Only the system header + knowledge base go here.
        The conversation-level system prompt is injected fresh at query time
        by CAGSystemFreshSession._build_prompt() so it never needs a cache rebuild.

        The marker "=== KNOWLEDGE BASE ===" is used by _build_prompt() to
        extract just the knowledge body when decoding the cached input_ids.
        Keep this marker consistent between both methods.
        """
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are the AI receptionist for Ask Novation, a business solutions company.\n"
            "You are warm, professional, and conversational.\n"
            "Use the knowledge base below to answer user questions accurately.\n"
            "\n"
            "=== KNOWLEDGE BASE ===\n"
            f"{knowledge_text}\n"
            "<|eot_id|>"
        )
        return prompt

    # ──────────────────────────────────────────────────────────────────────────
    # Startup / shutdown cleanup (synchronize IS acceptable here)
    # ──────────────────────────────────────────────────────────────────────────

    def _cleanup_memory(self):
        """Heavy cleanup — only call at startup or shutdown, not per-query."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()