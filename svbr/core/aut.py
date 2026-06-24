from __future__ import annotations

import re


HEADER_RE = re.compile(r"^\s*des\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$")


def parse_aut_header(line: str) -> tuple[int, int, int]:
    match = HEADER_RE.match(line.strip())
    if not match:
        raise ValueError(f"Invalid AUT header: {line.strip()}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def parse_aut_transition(line: str) -> tuple[int, str, int] | None:
    text = line.strip()
    if not text or not text.startswith("(") or not text.endswith(")"):
        return None
    body = text[1:-1]
    first_comma = body.find(",")
    if first_comma < 0:
        return None
    src = int(body[:first_comma].strip())
    rest = body[first_comma + 1 :].strip()
    if rest.startswith('"'):
        last_quote = rest.rfind('"')
        if last_quote <= 0:
            raise ValueError(f"Invalid quoted AUT label: {line.strip()}")
        action = rest[1:last_quote]
        tail = rest[last_quote + 1 :].strip()
        if not tail.startswith(","):
            raise ValueError(f"Missing destination after AUT label: {line.strip()}")
        dst = int(tail[1:].strip())
    else:
        action_text, dst_text = rest.rsplit(",", 1)
        action = action_text.strip()
        dst = int(dst_text.strip())
    return src, action, dst
