"""Name normalisation helpers shared by the character animation modules."""

from __future__ import annotations

import re


def slugify(value: str) -> str:
    """Return a filesystem-friendly representation of ``value``."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "item"
