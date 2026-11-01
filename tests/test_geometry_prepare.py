from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from datacompose.geometry_prepare import _dimensions, _resolve_asset_path


class GeometryPrepareTest(unittest.TestCase):
    def test_inventory_path_is_bounded_fallback_for_legacy_glbs_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            object_root = Path(temporary)
            asset = object_root / "000-000" / "object.glb"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"glb")
            resolved, relative = _resolve_asset_path(
                {
                    "canonical_path": "glbs/000-000/object.glb",
                    "inventory_path": "000-000/object.glb",
                },
                object_root,
            )
            self.assertEqual(resolved, asset.resolve())
            self.assertEqual(relative, "000-000/object.glb")

    def test_non_finite_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite positive"):
            _dimensions([1.0, float("nan"), 1.0])


if __name__ == "__main__":
    unittest.main()
