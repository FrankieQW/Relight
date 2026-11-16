#!/usr/bin/env python
"""Smoke-test ``blender/probe_scene.py`` end to end, in the target Blender.

The probe baker is the one part of the pipeline that cannot be unit tested: it
needs a real ``BVHTree`` and a real ``.blend``.  Everything else about the probe
(queries, planning, persistence) is covered by ``tests/test_scene_probe.py``.

So before trusting a new Blender build, run::

    python tools/probe_smoke_test.py --blender-bin /opt/blender-4.5/blender

It builds a tiny hollow room with a table and a cup, bakes it, and checks the
numbers against exact analytic values.  Nothing is left behind unless ``--keep``
is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


PACKAGE_DIR = Path(__file__).resolve().parents[1]

#: Runs inside Blender: build a hollow 10 x 10 x 3 room, a table, a cup.
BUILD_SCENE = r'''
import sys

import bpy

target = sys.argv[sys.argv.index("--out") + 1]
bpy.ops.wm.read_factory_settings(use_empty=True)


def quad(name, corners):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(corners, [], [(0, 1, 2, 3)])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj["lc_scene_object_id"] = "shell"


def box(name, low, high, node_id):
    (x0, y0, z0), (x1, y1, z1) = low, high
    verts = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 1, 2, 3), (7, 6, 5, 4), (0, 4, 5, 1),
        (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj["lc_scene_object_id"] = node_id


quad("Floor", [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0)])
quad("Ceil", [(0, 0, 3), (0, 10, 3), (10, 10, 3), (10, 0, 3)])
quad("WallY0", [(0, 0, 0), (0, 0, 3), (10, 0, 3), (10, 0, 0)])
quad("WallY10", [(0, 10, 0), (10, 10, 0), (10, 10, 3), (0, 10, 3)])
quad("WallX0", [(0, 0, 0), (0, 10, 0), (0, 10, 3), (0, 0, 3)])
quad("WallX10", [(10, 0, 0), (10, 0, 3), (10, 10, 3), (10, 10, 0)])
box("Table", (4, 4, 0.72), (5, 5, 0.78), "table")
box("Cup", (4.4, 4.4, 0.78), (4.6, 4.6, 0.98), "cup")
bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath=target)
print("[smoke] built", target, flush=True)
'''


def _run(command: list[str], label: str) -> None:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-25:])
        raise SystemExit(f"{label} failed (exit {completed.returncode}):\n{tail}")
    for line in completed.stdout.splitlines():
        if line.startswith("[smoke]") or line.startswith("[probe]"):
            print("   ", line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--keep", action="store_true", help="keep the temp files")
    args = parser.parse_args()

    blender = args.blender_bin.split()
    work = Path(tempfile.mkdtemp(prefix="probe_smoke_"))
    try:
        build_script = work / "build_scene.py"
        build_script.write_text(BUILD_SCENE, encoding="utf-8")
        blend = work / "room.blend"
        print(f"[1/3] building a test scene in {work}")
        _run(
            [
                *blender,
                "--background",
                "--factory-startup",
                "--python",
                str(build_script),
                "--",
                "--out",
                str(blend),
            ],
            "scene build",
        )
        if not blend.is_file():
            raise SystemExit(f"Blender did not write {blend}")

        job = {
            "scene_id": "probe_smoke",
            "blend": str(blend),
            "output": str(work / "room.probe.npz"),
            "summary": str(work / "room.summary.json"),
            "resolution": args.resolution,
            "max_voxels": 4_000_000,
            "clearance_cap": 1.0,
            "padding": 0.1,
        }
        job_path = work / "job.json"
        job_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        print("[2/3] baking")
        _run(
            [
                *blender,
                "--background",
                "--factory-startup",
                "--python",
                str(PACKAGE_DIR / "blender" / "probe_scene.py"),
                "--",
                "probe",
                "--job",
                str(job_path),
            ],
            "probe bake",
        )

        print("[3/3] checking the field against exact values")
        _check(Path(job["output"]), args.resolution)
        print("\nPASS: probe_scene.py bakes a correct clearance field")
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            import shutil

            shutil.rmtree(work, ignore_errors=True)


def _check(path: Path, resolution: float) -> None:
    import numpy as np

    if str(PACKAGE_DIR.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE_DIR.parent))
    from datacompose.probes import ProbeVolume

    probe = ProbeVolume.load(path)
    # The resolution is coarsened to fit the voxel budget, so compare against
    # the resolution that was actually used.
    step = probe.resolution
    tolerance = 2.0 * step

    failures: list[str] = []

    def expect(label: str, value: float, target: float, tol: float) -> None:
        ok = abs(value - target) <= tol
        print(f"    {label:44s} {value:8.3f} (expect {target:.3f} +-{tol:.3f}) {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(label)

    def clearance(point) -> float:
        return float(probe.clearance_at([point])[0])

    if set(probe.node_ids) != {"shell", "table", "cup"}:
        failures.append(f"node ids: {probe.node_ids}")
    # ``clearance_at`` subtracts the distance to the voxel centre, so even a
    # capped reading lands one voxel-diagonal below the cap.  That is the point:
    # the query is a guaranteed lower bound, never an over-estimate.
    expect("centre of the room (capped)", clearance([5.0, 8.0, 1.5]), 1.0, tolerance)
    expect("next to the west wall", clearance([0.0, 5.0, 1.5]), 0.0, tolerance)
    expect("just above the table top", clearance([4.5, 4.5, 0.79]), 0.0, tolerance)
    expect("mid-air, 0.6 m from the table", clearance([3.4, 4.5, 0.78]), 0.6, tolerance)

    # The camera must never be told there is more free space than there is.
    if clearance([4.5, 4.5, 0.79]) > clearance([3.4, 4.5, 0.78]):
        failures.append("clearance is not monotonic away from the table")

    depth, node = probe.cast(
        [1.0, 5.0, 1.5], np.array([[1.0, 0.0, 0.0]]), max_distance=45.0, threshold=0.1
    )
    expect("ray from the west wall hits the east wall", float(depth[0]), 9.0, tolerance + 0.15)
    if probe.node_ids[int(node[0])] != "shell":
        failures.append(f"ray should hit the shell, got {probe.node_ids[int(node[0])]}")

    depth, node = probe.cast(
        [4.5, 4.5, 1.5], np.array([[0.0, 0.0, -1.0]]), max_distance=45.0, threshold=0.1
    )
    expect("ray downwards hits the cup", float(depth[0]), 0.52, tolerance + 0.1)
    if probe.node_ids[int(node[0])] != "cup":
        failures.append(f"downward ray should hit the cup, got {probe.node_ids[int(node[0])]}")

    # Floor support: a free column in the room must stand on z = 0.
    floor = probe.support_z(0.2)
    cell = np.floor((np.array([2.0, 2.0, 1.5]) - probe.origin) / step).astype(int)
    value = float(floor[int(cell[0]), int(cell[1]), int(cell[2])])
    expect("floor height under the camera", value, 0.0, tolerance)

    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures))


if __name__ == "__main__":
    main()
