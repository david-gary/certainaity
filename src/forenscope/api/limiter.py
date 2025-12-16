"""slowapi rate-limiter singleton for the ForenScope API.

The Limiter is instantiated once at module level and shared between
``main.py`` (which registers it on ``app.state`` and wires the exception
handler) and ``routes.py`` (which applies per-endpoint decorators).
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
