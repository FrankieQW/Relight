#!/usr/bin/env bash
# Remove the object / LLM-annotation / composition-era files from the checkout.
#
# This script exists because the scene-only rewrite was prepared in an
# environment where file deletion was blocked by a Windows ACL; run it once on a
# normal checkout and commit the result:
#
#     bash scripts/remove_legacy_object_pipeline.sh          # dry run
#     bash scripts/remove_legacy_object_pipeline.sh --apply   # delete
#
# Everything it removes is either replaced (render_compositions.py →
# render_scenes.py, blender/render_composition.py → blender/render_scene.py) or
# belongs to the abandoned object pipeline. Nothing else is touched.
set -euo pipefail

cd "$(dirname "$0")/.."

APPLY=0
if [ "${1:-}" = "--apply" ]; then
  APPLY=1
fi

FILES=(
  # object inventory / geometry / LLM pairing
  object_index.py
  geometry_prepare.py
  llm_annotate.py
  render_jobs.py
  prepare_objects.py
  prepare_geometry.py
  annotate_construction.py
  build_render_jobs.py
  # object-scene composition scheduler + Blender worker (replaced by scene-only)
  render_compositions.py
  blender/render_composition.py
  # object-only configs
  configs/category_dimensions.yaml
  configs/object_overrides.yaml
  # tests for the removed modules
  tests/test_render_jobs.py
  tests/test_geometry_prepare.py
  tests/test_render_compositions.py
  tests/test_composition_validator.py
  tests/blender_composition_focus.py
)

missing=0
for file in "${FILES[@]}"; do
  if [ -e "$file" ]; then
    if [ "$APPLY" -eq 1 ]; then
      git rm -q -- "$file" 2>/dev/null || rm -f -- "$file"
      echo "removed  $file"
    else
      echo "would remove  $file"
    fi
  else
    missing=$((missing + 1))
  fi
done

echo
if [ "$APPLY" -eq 1 ]; then
  echo "Done. Remaining Python modules:"
  find . -name '*.py' -not -path './.git/*' -not -path '*/__pycache__/*' | sort
  echo
  echo "Next: python -m unittest discover -s tests -p 'test_*.py'"
else
  echo "$missing file(s) already absent. Re-run with --apply to delete."
fi
