"""
test_latency.py — Unit tests for gateway/latency.py
  • TurnLatency: finalize, to_report
  • LatencyTracker: lifecycle (new_turn → events → complete_turn), session_summary

Run:
    pytest tests/test_latency.py -v
"""

import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from latency import TurnLatency, LatencyTracker, _r, _latency_color, _bar  # noqa


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper Function Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_round_none(self):
        assert _r(None) is None

    def test_round_value(self):
        assert _r(123.456) == 123.5

    def test_latency_color_green(self):
        assert "🟢" in _latency_color(100)

    def test_latency_color_yellow(self):
        assert "🟡" in _latency_color(300)

    def test_latency_color_orange(self):
        assert "🟠" in _latency_color(500)

    def test_latency_color_red(self):
        assert "🔴" in _latency_color(1000)

    def test_bar_full(self):
        bar = _bar(100, 100, w=10)
        assert bar == "█" * 10

    def test_bar_empty(self):
        bar = _bar(0, 100, w=10)
        assert bar == "░" * 10

    def test_bar_zero_max(self):
        bar = _bar(50, 0, w=10)
        assert bar == "░" * 10


# ═══════════════════════════════════════════════════════════════════════════════
#  TurnLatency Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTurnLatency:

    def test_default_values(self):
        t = TurnLatency()
        assert t.turn_id == ""
        assert t.barge_in is False
        assert t.total_tokens == 0

    def test_finalize_computes_stt_latency(self):
        t = TurnLatency()
        t.stt_first_word_ts = 100.0
        t.stt_segment_ts = 100.15
        t.finalize()
        assert t.stt_latency_ms == pytest.approx(150.0, abs=0.1)

    def test_finalize_computes_cag_first_token(self):
        t = TurnLatency()
        t.query_sent_ts = 100.0
        t.first_token_ts = 100.08
        t.finalize()
        assert t.cag_first_token_ms == pytest.approx(80.0, abs=0.1)

    def test_finalize_computes_e2e(self):
        t = TurnLatency()
        t.stt_first_word_ts = 100.0
        t.tts_audio_start_ts = 101.2
        t.finalize()
        assert t.e2e_ms == pytest.approx(1200.0, abs=0.1)

    def test_finalize_with_missing_timestamps(self):
        t = TurnLatency()
        t.finalize()  # should not crash
        assert t.stt_latency_ms is None
        assert t.e2e_ms is None

    def test_to_report_returns_dict(self):
        t = TurnLatency(turn_id="t1", query_text="hello")
        report = t.to_report()
        assert report["turn_id"] == "t1"
        assert report["query"] == "hello"
        assert "stt_latency_ms" in report
        assert "e2e_ms" in report

    def test_to_report_truncates_query(self):
        t = TurnLatency(query_text="x" * 200)
        report = t.to_report()
        assert len(report["query"]) <= 80


# ═══════════════════════════════════════════════════════════════════════════════
#  LatencyTracker Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatencyTracker:

    def test_init(self):
        lt = LatencyTracker(sid="test-session")
        assert lt.sid == "test-session"
        assert lt.current is None
        assert lt.history == []

    def test_new_turn_creates_turn(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "hello")
        assert lt.current is not None
        assert lt.current.turn_id == "t1"
        assert lt.current.query_text == "hello"

    def test_on_stt_first_word(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test")
        lt.on_stt_first_word()
        assert lt.current.stt_first_word_ts is not None

    def test_on_stt_first_word_only_once(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test")
        lt.on_stt_first_word()
        first = lt.current.stt_first_word_ts
        time.sleep(0.01)
        lt.on_stt_first_word()
        assert lt.current.stt_first_word_ts == first  # not updated

    def test_on_stt_segment(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test")
        lt.on_stt_segment()
        assert lt.current.stt_segment_ts is not None

    def test_on_query_sent(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test")
        lt.on_query_sent()
        assert lt.current.query_sent_ts is not None

    def test_on_first_token(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test")
        lt.on_query_sent()
        time.sleep(0.01)
        lt.on_first_token()
        assert lt.current.first_token_ts is not None
        assert lt.current.cag_first_token_ms is not None
        assert lt.current.cag_first_token_ms > 0

    def test_on_token_increments(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test")
        lt.on_token()
        lt.on_token()
        lt.on_token()
        assert lt.current.total_tokens == 3

    def test_complete_turn_returns_report(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "test query")
        lt.on_stt_first_word()
        lt.on_stt_segment()
        lt.on_query_sent()
        lt.on_first_token()
        report = lt.complete_turn()
        assert report is not None
        assert report["turn_id"] == "t1"
        assert lt.current is None  # cleared after complete

    def test_complete_turn_appends_to_history(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "q1")
        lt.complete_turn()
        lt.new_turn("t2", "q2")
        lt.complete_turn()
        assert len(lt.history) == 2

    def test_complete_turn_no_current_returns_none(self):
        lt = LatencyTracker(sid="s1")
        assert lt.complete_turn() is None

    def test_full_turn_lifecycle(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "How can I help?")
        lt.on_stt_first_word()
        time.sleep(0.01)
        lt.on_stt_segment()
        lt.on_query_sent()
        time.sleep(0.01)
        lt.on_first_token()
        lt.on_token()
        lt.on_token()
        lt.on_tts_chunk_sent("Hello there!")
        time.sleep(0.01)
        lt.on_tts_audio_start()
        lt.on_tts_chunk_complete(50.0, 45.0, 0.5)
        report = lt.complete_turn()
        assert report["stt_latency_ms"] is not None
        assert report["cag_first_token_ms"] is not None
        assert report["tts_synth_ms"] is not None
        assert report["e2e_ms"] is not None
        assert report["total_tokens"] == 2

    def test_session_summary_empty(self):
        lt = LatencyTracker(sid="s1")
        summary = lt.session_summary()
        assert summary["turns"] == 0

    def test_session_summary_with_turns(self):
        lt = LatencyTracker(sid="s1")

        for i in range(3):
            lt.new_turn(f"t{i}", f"query {i}")
            lt.on_stt_first_word()
            lt.on_stt_segment()
            lt.on_query_sent()
            lt.on_first_token()
            lt.on_tts_chunk_sent("text")
            lt.on_tts_audio_start()
            lt.complete_turn()

        summary = lt.session_summary()
        assert summary["turns"] == 3
        assert "stt" in summary
        assert "cag" in summary
        assert "e2e" in summary
        assert "min" in summary["e2e"]
        assert "p95" in summary["e2e"]

    def test_all_reports(self):
        lt = LatencyTracker(sid="s1")
        lt.new_turn("t1", "q1")
        lt.complete_turn()
        reports = lt.all_reports()
        assert len(reports) == 1
        assert reports[0]["turn_id"] == "t1"
