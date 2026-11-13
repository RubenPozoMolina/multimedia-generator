"""Discovery of externally supplied character reference images.

Users can drop a folder with reference photographs and a ``characters.json``
manifest next to the screenplay.  The pipeline then uses those images instead
of generating the character sheets with the text-to-image model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

from utils.character_animation.naming import slugify

DEFAULT_INPUT_DIR = "input"
MANIFEST_NAME = "characters.json"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class CharacterInputLibrary:
    """Map character names to reference images supplied by the user.

    The library understands the manifest produced by the ``input`` folder in
    the examples (a list of objects with ``name`` and ``path``), a mapping of
    names to paths, and plain directories of images named after each character.
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        screenplay_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ):
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.screenplay_dir = Path(screenplay_dir) if screenplay_dir is not None else Path(".")
        self.logger = logger or logging.getLogger(__name__)
        self._references = self._discover_references()

    @classmethod
    def from_screenplay(
        cls,
        screenplay_path: str | Path,
        data: dict[str, Any],
        logger: logging.Logger | None = None,
    ) -> "CharacterInputLibrary":
        """Build a library from the screenplay location and its ``input`` folder.

        A global ``character_input`` / ``characters_input`` field can point to
        either a directory or a JSON manifest.  When it is absent, an ``input``
        folder next to the screenplay is detected automatically.
        """
        screenplay_dir = Path(screenplay_path).parent
        configured = data.get("character_input") or data.get("characters_input")

        if configured:
            candidate = Path(str(configured)).expanduser()
            if not candidate.is_absolute():
                candidate = screenplay_dir / candidate
            if candidate.suffix.lower() == ".json":
                return cls(
                    base_dir=candidate.parent,
                    manifest_path=candidate,
                    screenplay_dir=screenplay_dir,
                    logger=logger,
                )
            return cls(
                base_dir=candidate,
                manifest_path=candidate / MANIFEST_NAME,
                screenplay_dir=screenplay_dir,
                logger=logger,
            )

        default_dir = screenplay_dir / DEFAULT_INPUT_DIR
        if not default_dir.is_dir():
            return cls(screenplay_dir=screenplay_dir, logger=logger)
        return cls(
            base_dir=default_dir,
            manifest_path=default_dir / MANIFEST_NAME,
            screenplay_dir=screenplay_dir,
            logger=logger,
        )

    @property
    def has_references(self) -> bool:
        """Whether at least one usable reference image was discovered."""
        return bool(self._references)

    def resolve(self, character_name: str) -> Path | None:
        """Return the supplied reference image for ``character_name``, if any."""
        return self._references.get(slugify(character_name))

    def _discover_references(self) -> dict[str, Path]:
        if self.base_dir is None:
            return {}

        references: dict[str, Path] = {}
        for name, raw_path in self._iter_manifest_entries():
            path = self._resolve_entry(name, raw_path)
            if path is None:
                if name:
                    self.logger.warning(
                        "Character input entry '%s' does not point to an existing image",
                        name,
                    )
                continue
            label = name or path.stem
            references[slugify(label)] = path
            self.logger.info("Using supplied character reference for %s: %s", label, path)

        for path in self._iter_images(self.base_dir):
            references.setdefault(slugify(path.stem), path)

        return references

    def _iter_manifest_entries(self) -> Iterator[tuple[str, str | None]]:
        if self.manifest_path is None or not self.manifest_path.is_file():
            return

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as manifest_file:
                data = json.load(manifest_file)
        except (OSError, ValueError) as error:
            self.logger.warning(
                "Could not read character input manifest %s: %s",
                self.manifest_path,
                error,
            )
            return

        entries = self._normalise_manifest(data)
        for entry in entries:
            if isinstance(entry, str):
                yield Path(entry).stem, entry
                continue
            if not isinstance(entry, dict):
                continue
            name = str(
                entry.get("name")
                or entry.get("character")
                or entry.get("id")
                or ""
            ).strip()
            raw_path = (
                entry.get("path")
                or entry.get("image")
                or entry.get("reference_image")
                or entry.get("file")
            )
            yield name, str(raw_path).strip() if raw_path else None

    @staticmethod
    def _normalise_manifest(data: Any) -> Sequence[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            nested = data.get("characters")
            if isinstance(nested, list):
                return nested
            if isinstance(nested, dict):
                return [
                    {"name": str(name), "path": str(path)}
                    for name, path in nested.items()
                ]
            if all(not isinstance(value, (dict, list)) for value in data.values()):
                return [
                    {"name": str(name), "path": str(path)}
                    for name, path in data.items()
                ]
        return []

    def _resolve_entry(self, name: str, raw_path: str | None) -> Path | None:
        for candidate in self._entry_candidates(name, raw_path):
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _entry_candidates(self, name: str, raw_path: str | None) -> Iterator[Path]:
        if raw_path:
            path = Path(raw_path).expanduser()
            if path.is_absolute():
                yield path
            else:
                yield self.screenplay_dir / path
                if self.base_dir is not None:
                    yield self.base_dir / path
                    yield self.base_dir / path.name

        if name and self.base_dir is not None:
            stem = slugify(name)
            for suffix in IMAGE_SUFFIXES:
                yield self.base_dir / f"{stem}{suffix}"

    @staticmethod
    def _iter_images(directory: Path) -> Iterator[Path]:
        if not directory.is_dir():
            return
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path
