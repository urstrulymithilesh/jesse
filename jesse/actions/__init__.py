"""Doing things on the computer: opening what he asks for, finding a file, media keys.

Split the same way the scheduler is: `parse` is pure and decides WHAT was asked,
`run` does it. See parse.py for why this is deterministic and not a tool-calling model.
"""

from jesse.actions.parse import (ActionCommand, FindCommand, MediaCommand, OpenCommand,
                                UnknownTarget, parse_action_command)
from jesse.actions.run import find_files, media_key, open_target

__all__ = [
    "ActionCommand", "FindCommand", "MediaCommand", "OpenCommand", "UnknownTarget",
    "parse_action_command", "find_files", "media_key", "open_target",
]
