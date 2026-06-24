from __future__ import annotations


def progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    done = max(0, min(done, total))
    filled = int(width * done / total)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def print_progress(stage: str, done: int, total: int, **fields) -> None:
    percent = 100.0 if total <= 0 else 100.0 * max(0, min(done, total)) / total
    details = " ".join(f"{key}={value}" for key, value in fields.items() if value not in {"", None})
    suffix = f" {details}" if details else ""
    print(f"[{stage}] {progress_bar(done, total)} {done}/{total} {percent:5.1f}%{suffix}", flush=True)
