"""Narrow recognition of explicit memory declarations, separate from recall questions."""

import re

_REMEMBER = re.compile(r"^\s*(?:please\s+)?remember\s+that\s+(.+?)\s*$", re.I | re.S)


def parse_remember_request(text: str) -> str | None:
    """Return the statement in 'remember that ...', never 'do you remember ...?'.

    Deliberately requires 'that': bare 'remember my birthday' can mean recall or
    a future reminder. Questions and empty declarations stay on existing routes.
    """
    match = _REMEMBER.fullmatch(text)
    if match is None:
        return None
    statement = match[1].strip()
    if statement.endswith("?") or not statement.strip(".! "):
        return None
    return statement
