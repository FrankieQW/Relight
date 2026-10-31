"""datacompose: LightConstruction data-synthesis package.

Everything that turns Objaverse objects and Bistro scenes into TokenLight
components and manifests lives here: the preparation pipeline
(``prepare-objects`` / ``prepare-scenes`` / ``annotate-construction`` /
``prepare-geometry`` / ``build-render-jobs`` / ``render``), the Blender-side
workers, the TokenLight render helpers, manifest builder and validator, the
stage-2 orchestration helpers, and the camera/image-plane coordinate module.

Coordinate conventions are documented in ``datacompose/GUID.md``.
"""

__version__ = "0.1.0"
