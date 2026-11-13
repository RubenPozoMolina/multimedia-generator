"""Build a single reference image for multi-character scenes.

Qwen-Image-Edit in the pinned diffusers version accepts one image at a time.
For scenes with several characters we therefore compose the primary reference
of each character into a horizontal board and ask the editor to turn that
board into one coherent scene.  This keeps the identity of every character
available to the model without changing the existing image models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image

logger = logging.getLogger(__name__)


class CharacterBoardBuilder:
    """Compose character references into a single conditioning image."""

    def __init__(
        self,
        gap: int = 24,
        panel_height: int = 512,
        background_color: tuple[int, int, int] = (236, 236, 236),
    ):
        self.gap = gap
        self.panel_height = panel_height
        self.background_color = background_color

    def build(
        self,
        character_names: Sequence[str],
        references: Mapping[str, str],
        output_path: str | Path,
    ) -> str:
        """Return a single image path containing all requested characters.

        For a single character the reference is returned untouched, avoiding an
        unnecessary resample.  For multiple characters a horizontal board is
        written to ``output_path``.
        """
        if not character_names:
            raise ValueError("Cannot build a character board without characters")

        missing = [name for name in character_names if name not in references]
        if missing:
            raise ValueError(f"Missing character references: {', '.join(missing)}")

        if len(character_names) == 1:
            return str(references[character_names[0]])

        destination = Path(output_path)
        # Rebuilding this lightweight board is cheap and prevents stale character
        # images from surviving after a supplied reference has been replaced.
        panels = [self._load_panel(references[name]) for name in character_names]
        width = sum(panel.width for panel in panels) + self.gap * (len(panels) + 1)
        height = self.panel_height + self.gap * 2
        board = Image.new("RGB", (width, height), self.background_color)

        offset_x = self.gap
        for panel in panels:
            board.paste(panel, (offset_x, self.gap))
            offset_x += panel.width + self.gap

        destination.parent.mkdir(parents=True, exist_ok=True)
        board.save(str(destination))
        logger.info("Character board saved to %s", destination)
        return str(destination)

    def _load_panel(self, reference_path: str | Path) -> Image.Image:
        with Image.open(reference_path) as source:
            image = source.convert("RGB")
        if image.height == self.panel_height:
            return image
        width = max(1, int(round(image.width * self.panel_height / image.height)))
        return image.resize(
            (width, self.panel_height), resample=Image.Resampling.LANCZOS
        )
