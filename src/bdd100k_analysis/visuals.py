"""Visualization helpers for BDD100K analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageDraw


def _resolve_image(image_root: Path, image_name: str) -> Path:
    """Resolve an image filename relative to the root directory."""
    return image_root / image_name


def draw_box_sample(
    image_path: Path,
    box: Dict[str, float],
    output_path: Path,
    label: str | None = None,
    color: str = "red",
) -> None:
    """Draw a single bounding box on an image and save it."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    coords = [
        float(box["x1"]),
        float(box["y1"]),
        float(box["x2"]),
        float(box["y2"]),
    ]
    draw.rectangle(coords, outline=color, width=3)
    if label:
        draw.text((coords[0] + 4, coords[1] + 4), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _iter_extremes(samples: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Iterate over largest/smallest samples."""
    for group in ("largest", "smallest"):
        for sample in samples.get(group, []):
            yield {"group": group, **sample}


def _iter_aspect(samples: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """Iterate over widest/tallest samples."""
    for group in ("widest", "tallest"):
        for sample in samples.get(group, []):
            yield {"group": group, **sample}


def render_extreme_samples(
    summary: Dict[str, Any],
    images_root: Path,
    output_dir: Path,
    limit_per_group: int = 5,
) -> List[Path]:
    """Render extreme samples for each class from summary stats."""
    rendered: List[Path] = []
    classes = summary.get("classes", {})

    for class_name, stats in classes.items():
        area_samples = list(_iter_extremes(stats.get("area", {})))[
            : 2 * limit_per_group
        ]
        aspect_samples = list(_iter_aspect(stats.get("aspect", {})))[
            : 2 * limit_per_group
        ]

        for sample in area_samples + aspect_samples:
            image_name = sample.get("image")
            if not image_name:
                continue
            image_path = _resolve_image(images_root, image_name)
            if not image_path.exists():
                continue
            group = sample.get("group", "sample")
            if group == "smallest":
                group = "small"
            elif group == "largest":
                group = "large"
            label = f"{class_name} {group}"
            output_path = output_dir / class_name / f"{group}_{image_name}"
            draw_box_sample(
                image_path,
                sample.get("box", {}),
                output_path,
                label=label,
            )
            rendered.append(output_path)
    return rendered
