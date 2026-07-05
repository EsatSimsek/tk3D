from __future__ import annotations

import time


class ProgressBar:
    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(int(total), 0)
        self.width = width
        self.start_time = time.perf_counter()
        self.last_value = 0
        self.print(0, extra="starting")

    def print(self, value: int, extra: str = "") -> None:
        self.last_value = min(max(int(value), 0), self.total) if self.total else max(int(value), 0)
        elapsed = max(time.perf_counter() - self.start_time, 1e-9)
        percent = (self.last_value / self.total * 100.0) if self.total else 100.0
        rate = self.last_value / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - self.last_value, 0)
        eta = remaining / rate if rate > 0 else 0.0
        filled = int(round(self.width * min(max(percent, 0.0), 100.0) / 100.0))
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f"  {extra}" if extra else ""
        print(
            f"\r      {self.label:<18} [{bar}] {percent:5.1f}%  "
            f"{self.last_value:>4}/{self.total:<4}  "
            f"speed {rate:4.2f}/s  "
            f"elapsed {_format_seconds(elapsed)}  "
            f"eta {_format_seconds(eta)}{suffix}",
            end="",
            flush=True,
        )

    def done(self, extra: str = "done") -> None:
        target = self.total if self.total else self.last_value
        self.print(target, extra=extra)
        print(flush=True)


def print_step(step: int, total: int, label: str) -> None:
    percent = step / total * 100.0 if total else 100.0
    print(f"[{step}/{total}] {label} ({percent:5.1f}%)", flush=True)


def _format_seconds(seconds: float) -> str:
    total = int(round(max(seconds, 0.0)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"
