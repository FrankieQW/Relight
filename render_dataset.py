#!/usr/bin/env python
"""Render TokenLight components for every planned scene camera.

Needs a Blender 4.5 binary and (by default) a CUDA GPU — see ``m4.render``.
No object assets and no ``OBJECT_ROOT``: each job renders an existing scene.

    python render_dataset.py --blender-bin /opt/blender-4.5/blender --workers 2
"""

from _launcher import run


if __name__ == "__main__":
    run("render")
