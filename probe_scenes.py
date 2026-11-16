#!/usr/bin/env python
"""Bake scene meshes into clearance probes (needs Blender).

    python probe_scenes.py --blender-bin /opt/blender-4.5/blender
    python probe_scenes.py --config sceneonly.yaml --no-resume
"""

from _launcher import run


if __name__ == "__main__":
    run("probe-scenes")
