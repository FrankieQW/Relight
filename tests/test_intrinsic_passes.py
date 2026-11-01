"""Pure-NumPy tests for the intrinsic passes and the material audit.

These run without Blender, a GPU or OpenCV so they can gate CI cheaply.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # the datacompose package directory
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datacompose import intrinsic  # noqa: E402
from datacompose.intrinsic import (  # noqa: E402
    albedo_quality,
    derive_irradiance,
    evaluate_material_audit,
    local_variance,
    valid_mask,
    world_normals_to_camera,
)

VALIDATOR_PATH = PACKAGE_ROOT / "tokenlight" / "validate_components.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("intrinsic_validate_components", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def temp_root():
    """A writable temp directory, tolerant of the sandbox's 0o700 temp issue."""
    try:
        base = Path(tempfile.mkdtemp())
        probe = base / "_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except OSError:
        base = PACKAGE_ROOT / f".intrinsic_tmp_{uuid.uuid4().hex}"
        base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _checker(height: int, width: int, low: float = 0.2, high: float = 0.8) -> np.ndarray:
    image = np.full((height, width, 3), low, dtype=np.float32)
    image[1::2, 1::2, :] = high
    return image


class IrradianceTest(unittest.TestCase):
    def test_division_recovers_e_pi(self) -> None:
        albedo = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        direct = np.array([[[0.3, 0.3, 0.3]]], dtype=np.float32)
        indirect = np.array([[[0.2, 0.2, 0.2]]], dtype=np.float32)
        irradiance, valid = derive_irradiance(albedo, direct, indirect)
        self.assertTrue(bool(valid[0, 0]))
        np.testing.assert_allclose(irradiance[0, 0], [1.0, 1.0, 1.0], rtol=1e-6)

    def test_low_albedo_pixels_are_zeroed_and_invalid(self) -> None:
        albedo = np.array([[[0.0005, 0.5, 0.5], [0.5, 0.5, 0.5]]], dtype=np.float32)
        direct = np.ones_like(albedo)
        indirect = np.zeros_like(albedo)
        irradiance, valid = derive_irradiance(albedo, direct, indirect, eps=1e-3)
        self.assertFalse(bool(valid[0, 0]))
        np.testing.assert_allclose(irradiance[0, 0], [0.0, 0.0, 0.0])
        self.assertTrue(bool(valid[0, 1]))

    def test_eps_is_inclusive_boundary(self) -> None:
        albedo = np.array([[[1e-3, 1e-3, 1e-3]]], dtype=np.float32)
        direct = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        indirect = np.zeros_like(direct)
        irradiance, valid = derive_irradiance(albedo, direct, indirect, eps=1e-3)
        self.assertTrue(bool(valid[0, 0]))
        np.testing.assert_allclose(irradiance[0, 0], [1000.0, 1000.0, 1000.0], rtol=1e-5)

    def test_non_positive_eps_is_rejected(self) -> None:
        image = np.zeros((1, 1, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            derive_irradiance(image, image, image, eps=0.0)


class NormalsTest(unittest.TestCase):
    def test_world_up_maps_to_camera_down(self) -> None:
        frame = SimpleNamespace(right=[1, 0, 0], up=[0, 1, 0], forward=[0, 0, 1])
        world = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        camera = world_normals_to_camera(world, frame)
        np.testing.assert_allclose(camera[0, 0], [0.0, -1.0, 0.0], atol=1e-6)

    def test_rotation_by_90_degrees(self) -> None:
        # Camera rotated 90 degrees about +Z: right=-Y, up=+X, forward=+Z.
        frame = SimpleNamespace(right=[0, -1, 0], up=[1, 0, 0], forward=[0, 0, 1])
        world = np.array([[[0.0, 1.0, 0.0]]], dtype=np.float32)
        camera = world_normals_to_camera(world, frame)
        # dot(world, right)=-1, dot(world, up)=0, dot(world, forward)=0 -> [-1, 0, 0]
        np.testing.assert_allclose(camera[0, 0], [-1.0, 0.0, 0.0], atol=1e-6)

    def test_output_is_unit_length_and_keeps_zero_background(self) -> None:
        frame = SimpleNamespace(right=[1, 0, 0], up=[0, 1, 0], forward=[0, 0, 1])
        world = np.array([[[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        camera = world_normals_to_camera(world, frame)
        self.assertAlmostEqual(float(np.linalg.norm(camera[0, 0])), 1.0, places=6)
        np.testing.assert_allclose(camera[0, 1], [0.0, 0.0, 0.0])


class ValidMaskTest(unittest.TestCase):
    def test_requires_albedo_above_eps_and_geometry(self) -> None:
        albedo = np.array(
            [[[0.5, 0.5, 0.5], [0.0001, 0.5, 0.5], [0.5, 0.5, 0.5]]], dtype=np.float32
        )
        normals = np.zeros_like(albedo)
        normals[0, 0] = [0.0, 0.0, 1.0]
        normals[0, 1] = [0.0, 0.0, 1.0]
        normals[0, 2] = [0.0, 0.0, 0.0]
        mask = valid_mask(albedo, normals, eps=1e-3)
        self.assertEqual(mask.tolist(), [[True, False, False]])


class AlbedoQualityTest(unittest.TestCase):
    def test_flat_image_has_no_texture(self) -> None:
        flat = np.full((8, 8, 3), 0.5, dtype=np.float32)
        quality = albedo_quality(flat)
        self.assertLess(quality["textured_pixel_ratio"], 0.02)
        self.assertEqual(quality["magenta_ratio"], 0.0)

    def test_checker_image_is_textured(self) -> None:
        quality = albedo_quality(_checker(8, 8))
        self.assertGreater(quality["textured_pixel_ratio"], 0.5)
        self.assertGreater(local_variance(_checker(8, 8)).max(), 1e-6)

    def test_magenta_placeholder_is_detected(self) -> None:
        magenta = np.zeros((8, 8, 3), dtype=np.float32)
        magenta[..., 0] = 1.0
        magenta[..., 2] = 1.0
        quality = albedo_quality(magenta)
        self.assertGreater(quality["magenta_ratio"], 0.99)


class MaterialAuditTest(unittest.TestCase):
    def test_healthy_report_passes(self) -> None:
        report = {"textured_materials": 4, "base_color_linked_ratio": 1.0, "missing_image_files": 0}
        self.assertEqual(evaluate_material_audit(report, require_textures=True), [])

    def test_low_ratio_is_a_problem(self) -> None:
        report = {"textured_materials": 4, "base_color_linked_ratio": 0.5, "missing_image_files": 0}
        problems = evaluate_material_audit(report, require_textures=True)
        self.assertTrue(problems)

    def test_missing_images_are_a_problem(self) -> None:
        report = {
            "textured_materials": 4,
            "base_color_linked_ratio": 1.0,
            "missing_image_files": 2,
            "missing_examples": ["a.dds", "b.dds"],
        }
        problems = evaluate_material_audit(report, require_textures=True)
        self.assertTrue(any("missing" in problem for problem in problems))

    def test_advisory_mode_never_reports_problems(self) -> None:
        report = {"textured_materials": 1, "base_color_linked_ratio": 0.0, "missing_image_files": 9}
        self.assertEqual(evaluate_material_audit(report, require_textures=False), [])


def _write_pass_set(
    root: Path,
    *,
    albedo: np.ndarray,
    normal: np.ndarray,
    valid: np.ndarray,
    irradiance: np.ndarray,
    quality: dict | None = None,
) -> dict:
    np.save(root / "albedo.npy", albedo.astype(np.float32))
    np.save(root / "normal_camera.npy", normal.astype(np.float32))
    np.save(root / "valid_mask.npy", valid)
    np.save(root / "irradiance.npy", irradiance.astype(np.float32))
    audit = {"textured_materials": 3, "base_color_linked_ratio": 1.0, "missing_image_files": 0}
    (root / "material_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return {
        "definition": {"eps": 1e-3, "albedo": "cycles_diffuse_color"},
        "albedo": "albedo.npy",
        "normal_camera": "normal_camera.npy",
        "valid_mask": "valid_mask.npy",
        "irradiance": {"ambient": "irradiance.npy"},
        "material_audit": "material_audit.json",
        "quality": quality
        or {"base_color_linked_ratio": 1.0, "textured_pixel_ratio": 1.0, "magenta_ratio": 0.0},
    }


class ValidatorPassesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = load_validator()

    def _base(self, height: int = 8, width: int = 8):
        albedo = _checker(height, width)
        normal = np.zeros((height, width, 3), dtype=np.float32)
        normal[..., 2] = 1.0
        valid = np.ones((height, width), dtype=bool)
        irradiance = np.full((height, width, 3), 0.5, dtype=np.float32)
        return albedo, normal, valid, irradiance

    def _run(self, albedo, normal, valid, irradiance, *, require_textures=True):
        with temp_root() as root:
            passes = _write_pass_set(
                root,
                albedo=albedo,
                normal=normal,
                valid=valid,
                irradiance=irradiance,
            )
            scene = {"id": "scene-test", "passes": passes}
            ambient = np.zeros(albedo.shape, dtype=np.float32)
            counts: dict = {}
            self.validator.validate_passes(
                root, scene, ambient, counts, require_textures=require_textures
            )
            return counts

    def test_legal_pass_set_passes(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        counts = self._run(albedo, normal, valid, irradiance)
        self.assertEqual(counts["pass_sets"], 1)
        self.assertEqual(counts.get("pass_warnings"), [])

    def test_missing_albedo_field_raises(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        with temp_root() as root:
            passes = _write_pass_set(
                root, albedo=albedo, normal=normal, valid=valid, irradiance=irradiance
            )
            passes.pop("albedo")
            scene = {"id": "scene-test", "passes": passes}
            with self.assertRaises(ValueError):
                self.validator.validate_passes(
                    root, scene, np.zeros(albedo.shape, np.float32), {}, require_textures=False
                )

    def test_shape_mismatch_raises(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        bad_normal = normal[..., :2]
        with self.assertRaises(ValueError):
            self._run(albedo, bad_normal, valid, irradiance)

    def test_non_unit_normal_raises(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        bad_normal = normal * 2.0
        with self.assertRaises(ValueError):
            self._run(albedo, bad_normal, valid, irradiance)

    def test_negative_irradiance_raises(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        bad_irradiance = irradiance.copy()
        bad_irradiance[0, 0, 0] = -0.1
        with self.assertRaises(ValueError):
            self._run(albedo, normal, valid, bad_irradiance)

    def test_magenta_albedo_raises_when_textures_required(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        magenta = np.zeros_like(albedo)
        magenta[..., 0] = 1.0
        magenta[..., 2] = 1.0
        with self.assertRaises(ValueError):
            self._run(magenta, normal, valid, irradiance, require_textures=True)

    def test_magenta_albedo_is_only_a_warning_when_not_required(self) -> None:
        albedo, normal, valid, irradiance = self._base()
        magenta = np.zeros_like(albedo)
        magenta[..., 0] = 1.0
        magenta[..., 2] = 1.0
        counts = self._run(magenta, normal, valid, irradiance, require_textures=False)
        self.assertEqual(counts["pass_sets"], 1)
        self.assertTrue(any("magenta" in warning for warning in counts["pass_warnings"]))


if __name__ == "__main__":
    unittest.main()
