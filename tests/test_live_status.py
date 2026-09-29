"""Validates _LiveStatus (em_fit's in-place-refreshing periodic status
block) in all three of its modes: a real Jupyter kernel, a plain
interactive terminal, and neither (captured/redirected output, where it
must not emit any escape codes at all). Can't exercise this by actually
running em_fit under pytest, since pytest's captured stdout isn't a TTY --
these mock sys.stdout/IPython directly instead."""
import io
import sys
import types
from unittest import mock

from pyem.emloop import _LiveStatus


class _FakeTTY(io.StringIO):
    def isatty(self):
        return True


def test_tty_mode_clears_exactly_the_preceding_block():
    buf = _FakeTTY()
    with mock.patch("sys.stdout", buf):
        status = _LiveStatus(enabled=True)
        assert status.mode == "tty"
        status.update("iter: 1\nbetas: [1 2]\nchange: 0.5")  # 3 lines
        status.update("iter: 2\nbetas: [3 4]\nchange: 0.1")  # 3 lines
        status.update("iter: 3\nbetas: [5 6]\nsigma: [7]\nchange: 0.01")  # 4 lines

    out = buf.getvalue()
    assert "\x1b[3A\x1b[Jiter: 2" in out
    assert "\x1b[3A\x1b[Jiter: 3" in out  # clears the preceding (3-line) block, not the new one


def test_plain_fallback_emits_no_escape_codes():
    buf = io.StringIO()  # plain StringIO: isatty() is False by default
    with mock.patch("sys.stdout", buf):
        status = _LiveStatus(enabled=True)
        assert status.mode == "plain"
        status.update("iter: 1\nchange: 0.5")
        status.update("iter: 2\nchange: 0.1")

    out = buf.getvalue()
    assert "\x1b" not in out
    assert out == "iter: 1\nchange: 0.5\niter: 2\nchange: 0.1\n"


def test_disabled_prints_nothing():
    buf = io.StringIO()
    with mock.patch("sys.stdout", buf):
        _LiveStatus(enabled=False).update("should not appear")
    assert buf.getvalue() == ""


def test_ipython_mode_uses_clear_output():
    fake_shell = type("ZMQInteractiveShell", (), {})()
    ipython_mod = types.ModuleType("IPython")
    ipython_mod.get_ipython = lambda: fake_shell
    display_mod = types.ModuleType("IPython.display")
    clear_calls = []
    display_mod.clear_output = lambda wait=True: clear_calls.append(wait)

    with mock.patch.dict(sys.modules, {"IPython": ipython_mod, "IPython.display": display_mod}):
        status = _LiveStatus(enabled=True)
        assert status.mode == "ipython"
        status.update("iter: 1")
        status.update("iter: 2")

    assert clear_calls == [True, True]
