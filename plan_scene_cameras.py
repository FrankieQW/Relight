#!/usr/bin/env python
"""Plan collision-free cameras inside existing scenes (scene-only synthesis).

Reads ``scene.json`` (``paths.scene_output`` by default) and writes
``scene_camera_jobs.jsonl`` plus a ``.summary.json`` next to it.  No Blender, no
GPU, no object assets: this is the camera half of the scene-only plan described
in ``docs/scene_only_render_feasibility.md``.

    python plan_scene_cameras.py --cameras-per-scene 32
    python plan_scene_cameras.py --scene-json data/scene/scene.json --output /tmp/cams.jsonl
"""

from _launcher import run


if __name__ == "__main__":
    run("plan-scene-cameras")
