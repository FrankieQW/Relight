#!/usr/bin/env python
"""Clear one failed render partial after reviewing its ``failure.json``.

    python clear_render_partial.py --job-id composition_0123456789abcdef0123
"""

from _launcher import run


if __name__ == "__main__":
    run("clear-render-partial")
