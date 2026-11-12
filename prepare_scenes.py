#!/usr/bin/env python
"""Prepare FBX scenes: normalised ``.blend`` files plus ``scene.json``.

    python prepare_scenes.py --blender-bin /opt/blender-4.5/blender
    python prepare_scenes.py --config sceneonly.yaml --no-resume
"""

from _launcher import run


if __name__ == "__main__":
    run("prepare-scenes")
