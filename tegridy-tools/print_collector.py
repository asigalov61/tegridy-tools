# print_collector.py  — requires Python 3.8+
# Project Los Angeles / Tegridy Code 2026
# Apache 2.0 / Version 1.0.0

from __future__ import annotations

import builtins
import contextlib
import pprint
import sys
import threading
from typing import Any, Iterator

"""
Simple example usage:

from print_collector import PrintCollector

print = PrintCollector(pretty=True)   # shadows the built-in in this module

print("Training started")
print({"epoch": 1, "loss": 0.123, "metrics": [0.9, 0.8]})

log = print.get()      # everything printed so far (ends with \n)
print.clear()          # start fresh
"""

class PrintCollector:
    """
    A drop-in print() replacement that accumulates all output into a string.

    Basic use:
        >>> p = PrintCollector()
        >>> p("Hello", "world")
        >>> p("x =", 42)
        >>> p.get()
        'Hello world\\nx = 42\\n'

    As an actual print replacement (shadows the built-in for this module):
        from print_collector import PrintCollector
        print = PrintCollector(pretty=True)

    With pretty=True, strings are printed as-is, while everything else
    (dicts, lists, dataclasses, ...) is rendered with pprint.pformat().

    Extras:
      * Thread-safe (every call is guarded by a lock).
      * Works as a stream (write/flush), so ``print(..., file=collector)``
        and ``contextlib.redirect_stdout(collector)`` work too.
      * ``with p.capture():`` temporarily patches the *global* print() so
        prints from anywhere in the process are captured. Also usable as
        a decorator: ``@p.capture()``.
      * ``echo=True`` tees everything to the real stdout for debugging.
      * If ``file=`` is explicitly passed to a call, the text is captured
        *and* forwarded to that file, so legacy code keeps working.
    """

    def __init__(
        self,
        *,
        pretty: bool = False,
        width: int = 80,
        sort_dicts: bool = False,
        echo: bool = False,
    ) -> None:
        self._chunks: list[str] = []
        self._lock = threading.RLock()
        self._pretty = pretty
        self._width = width
        self._sort_dicts = sort_dicts   # False keeps dict insertion order
        self._echo = echo

    # ------------------------------------------------------------------ #
    # The print() replacement itself                                     #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        *objects: Any,
        sep: str | None = " ",
        end: str | None = "\n",
        file: Any = None,
        flush: bool = False,
    ) -> None:
        """Exactly like built-in print(), but the output is accumulated here."""
        if sep is None:
            sep = " "
        if end is None:
            end = "\n"

        if self._pretty:
            parts = [o if isinstance(o, str) else self.pformat(o) for o in objects]
        else:
            parts = [str(o) for o in objects]
        text = sep.join(parts) + end

        with self._lock:
            self._chunks.append(text)

        # Legacy compatibility: also honor an explicitly requested file.
        if file is not None and file is not self:
            file.write(text)
            if flush:
                file.flush()

        # Optional tee to the real stdout (never double-writes).
        if self._echo and sys.stdout is not self and file is not sys.stdout:
            sys.stdout.write(text)
            sys.stdout.flush()

    def pformat(self, obj: Any) -> str:
        """Pretty-print a single object to a string, on demand."""
        return pprint.pformat(obj, width=self._width, sort_dicts=self._sort_dicts)

    # ------------------------------------------------------------------ #
    # Stream protocol (lets this object pose as sys.stdout)              #
    # ------------------------------------------------------------------ #
    def write(self, s: str) -> int:
        if not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not {type(s).__name__}")
        with self._lock:
            self._chunks.append(s)
        return len(s)

    def flush(self) -> None:
        """No-op, for stream compatibility."""

    def close(self) -> None:
        """No-op, for file-object compatibility."""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # Reading / clearing the accumulated text                            #
    # ------------------------------------------------------------------ #
    def get(self, *, clear: bool = False) -> str:
        """Return everything collected so far; clear the buffer with clear=True."""
        with self._lock:
            text = "".join(self._chunks)
            if clear:
                self._chunks.clear()
        return text

    def getvalue(self, *, clear: bool = False) -> str:
        """Alias for get() — io.StringIO-compatible name."""
        return self.get(clear=clear)

    def clear(self) -> None:
        """Discard everything collected so far."""
        with self._lock:
            self._chunks.clear()

    @property
    def text(self) -> str:
        """Everything collected so far (non-destructive)."""
        return self.get()

    def __str__(self) -> str:
        return self.get()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(c) for c in self._chunks)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(pretty={self._pretty}, {len(self)} chars buffered)"

    # ------------------------------------------------------------------ #
    # Capture the *global* print() for a block of code                   #
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def capture(self, *, stdout: bool = False) -> Iterator["PrintCollector"]:
        """
        Temporarily replace built-in print() with this collector:

            with p.capture():
                print("captured!")          # goes into p, not stdout
                some_library_function()     # its prints are captured too
            text = p.get()

        With stdout=True, direct writes to sys.stdout are captured as well.
        """
        original_print = builtins.print
        builtins.print = self  # type: ignore[assignment]
        try:
            if stdout:
                with contextlib.redirect_stdout(self):
                    yield self
            else:
                yield self
        finally:
            builtins.print = original_print


if __name__ == "__main__":
    plain = PrintCollector()
    pretty = PrintCollector(pretty=True, width=40)

    user = {"name": "Ada Lovelace", "year": 1815,
            "languages": ["Analytical Engine", "Notes"], "pioneer": True}

    plain("Computing pioneer:", user)
    pretty("Computing pioneer:", user)

    print("=== plain ===")
    print(plain.get(), end="")
    print("=== pretty ===")
    print(pretty.get(), end="")

    with plain.capture():               # captures the real global print
        print("Hello from built-in print()")
        print("sqrt(2) =", 2 ** 0.5)
    print("=== captured ===")
    print(plain.get(clear=True), end="")

    import contextlib
    with contextlib.redirect_stdout(plain):   # stream-style works too
        print("redirect_stdout works")
    assert plain.get(clear=True) == "redirect_stdout works\n"
    print("All good.")