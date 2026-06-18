"""Assemble per-forecast-hour PNG frames into an animated GIF."""

from __future__ import annotations

from pathlib import Path

from ..config import PRODUCTS


def make_gif(frame_paths: list[str | Path], out_path: str | Path | None = None,
             duration_ms: int = 500) -> str:
    """Combine ordered PNG frames into a looping GIF (via Pillow)."""
    from PIL import Image

    if not frame_paths:
        raise ValueError("no frames to animate")
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    out_path = out_path or (PRODUCTS / "animation.gif")
    frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0,
                   duration=duration_ms, optimize=True)
    return str(out_path)
