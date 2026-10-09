"""Keep a plain-text copy of a child's terminal display as it redraws."""

from __future__ import annotations

import time
from typing import TextIO


class TerminalLog:
    def __init__(self, log: TextIO, *, snapshots: bool = False) -> None:
        self.log = log
        self.snapshots = snapshots
        self.rows = [""]
        self.offsets = [log.tell()]
        self.row = 0
        self.column = 0
        self.escape = ""
        self.dirty: int | None = None
        self.last_snapshot = 0.0
        self.last_printed = ""

    def _dirty(self) -> None:
        self.dirty = self.row if self.dirty is None else min(self.dirty, self.row)

    def _move(self, amount: int) -> None:
        self.row = max(0, self.row + amount)
        while self.row >= len(self.rows):
            self.rows.append("")
            self.offsets.append(self.offsets[-1])

    def _control(self, sequence: str) -> None:
        if not sequence.startswith("\x1b["):
            return
        code = sequence[-1]
        numbers = sequence[2:-1]
        if numbers.startswith("?"):
            return
        first = numbers.split(";")[0]
        count = int(first) if first.isdigit() else 1
        if code == "A":
            self._move(-count)
        elif code == "B":
            self._move(count)
        elif code == "C":
            self.column += count
        elif code == "D":
            self.column = max(0, self.column - count)
        elif code == "E":
            self._move(count)
            self.column = 0
        elif code == "F":
            self._move(-count)
            self.column = 0
        elif code == "G":
            self.column = max(0, count - 1)
        elif code in {"H", "f"}:
            parts = numbers.split(";")
            target_row = int(parts[0]) if parts[0].isdigit() else 1
            target_column = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            self._move(target_row - 1 - self.row)
            self.column = max(0, target_column - 1)
        elif code == "K":
            value = self.rows[self.row]
            if first == "1":
                self.rows[self.row] = " " * self.column + value[self.column:]
            elif first == "2":
                self.rows[self.row] = ""
                self.column = 0
            else:
                self.rows[self.row] = value[: self.column]
            self._dirty()

    def feed(self, chunk: str) -> None:
        for char in chunk:
            if self.escape:
                self.escape += char
                if len(self.escape) > 64 or ("@" <= char <= "~" and len(self.escape) > 2):
                    self._control(self.escape)
                    self.escape = ""
                continue
            if char == "\x1b":
                self.escape = char
            elif char == "\r":
                self.column = 0
            elif char == "\n":
                if self.snapshots:
                    self._print_row(self.rows[self.row])
                self._move(1)
                self.column = 0
                self._dirty()
            elif char == "\b":
                self.column = max(0, self.column - 1)
            elif char == "\t":
                self.column = (self.column // 8 + 1) * 8
            elif char >= " ":
                value = self.rows[self.row]
                if self.column > len(value):
                    value += " " * (self.column - len(value))
                self.rows[self.row] = value[: self.column] + char + value[self.column + 1 :]
                self.column += 1
                self._dirty()
        self.flush()
        if self.snapshots and self.rows[self.row].strip():
            now = time.monotonic()
            if now - self.last_snapshot >= 5 or "100%" in self.rows[self.row]:
                self._print_row(self.rows[self.row])
                self.last_snapshot = now

    def _print_row(self, value: str) -> None:
        value = value.rstrip()
        if value and value != self.last_printed:
            print(value, flush=True)
            self.last_printed = value

    def flush(self) -> None:
        if self.dirty is None:
            return
        start = self.dirty
        self.log.seek(self.offsets[start])
        self.log.truncate()
        for index in range(start, len(self.rows)):
            self.offsets[index] = self.log.tell()
            self.log.write(self.rows[index].rstrip())
            if index < len(self.rows) - 1:
                self.log.write("\n")
        self.log.flush()
        self.dirty = None

    def finish(self) -> None:
        if self.snapshots:
            self._print_row(self.rows[self.row])
        if self.rows[-1]:
            self.log.seek(0, 2)
            self.log.write("\n")
            self.log.flush()
