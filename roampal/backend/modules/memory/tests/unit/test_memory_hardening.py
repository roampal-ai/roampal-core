"""
v0.5.9 Item 5: MemoryError must leave a trace before the process exits.
Unit-level coverage of the detection + logging core (the heartbeat emission
and child-process capture behavior are exercised by the smoke tests recorded
in IMPLEMENTATION_TASKS.md and remain manual gates).
"""

import logging

from roampal.backend.modules.memory.search_service import (
    _contains_memory_error,
    log_memory_error,
)


class TestMemoryErrorDetection:
    def test_bare_memory_error(self):
        assert _contains_memory_error(MemoryError()) is True

    def test_unrelated_error(self):
        assert _contains_memory_error(RuntimeError("nope")) is False

    def test_wrapped_in_exception_group(self):
        # Structure-based check: any object with .exceptions is traversed,
        # so this works on 3.10 whether or not ExceptionGroup is importable.
        class FakeGroup(Exception):
            exceptions = (RuntimeError("x"), MemoryError("real"))

        assert _contains_memory_error(FakeGroup()) is True

    def test_nested_group(self):
        class FakeGroup(Exception):
            def __init__(self, *excs):
                self.exceptions = excs

        inner = FakeGroup(RuntimeError("a"))
        outer = FakeGroup(inner, RuntimeError("b"))
        assert _contains_memory_error(outer) is False

        outer_with_mem = FakeGroup(inner, MemoryError("deep"))
        assert _contains_memory_error(outer_with_mem) is True


class TestMemoryErrorLogging:
    def test_logs_fatal_record(self, caplog):
        logger = logging.getLogger("roampal.test.oom")
        with caplog.at_level(logging.ERROR, logger="roampal.test.oom"):
            log_memory_error(logger, MemoryError("commit limit"))
        assert any(
            "out-of-memory" in r.message and r.levelno == logging.ERROR
            for r in caplog.records
        )

    def test_exc_info_attached(self, caplog):
        logger = logging.getLogger("roampal.test.oom2")
        err = MemoryError("traceback needed")
        with caplog.at_level(logging.ERROR, logger="roampal.test.oom2"):
            log_memory_error(logger, err)
        assert any(r.exc_info is not None for r in caplog.records)

    def test_stderr_fallback_when_log_fails(self, capsys):
        class BrokenLogger:
            def error(self, *a, **kw):
                raise RuntimeError("logging is broken too")

        # Must never raise — falls back to stderr write.
        log_memory_error(BrokenLogger(), MemoryError("last words"))
        captured = capsys.readouterr()
        assert "FATAL MemoryError" in captured.err
