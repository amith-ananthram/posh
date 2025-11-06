"""Utilities for visualizing DOCENT annotations and PoSh granular scores.

This module exposes helpers for generating a side-by-side visualization of
an image, its reference description, and a candidate generation.  Spans that
have been annotated as *mistakes* in the generation are highlighted in red,
while spans that represent *omissions* in the reference are highlighted in
blue.  The visualization can be displayed interactively or saved to disk.

Example
-------
```python
from pathlib import Path
from PIL import Image

from posh.visualizer.visualize import create_visualization, save_visualization

image = Image.open(Path("example.jpg"))
reference = "A woman in a white dress holds a golden chalice."
generation = "A woman in a blue dress holds a silver cup."
mistakes = [{"start": 16, "end": 25}, {"start": 38, "end": 48}]
omissions = [{"start": 24, "end": 39}]

fig = create_visualization(image, reference, generation, mistakes, omissions)
fig.show()  # or plt.show()

save_visualization(
    image,
    reference,
    generation,
    mistakes,
    omissions,
    Path("visualization.png"),
)
```
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SpanInput = Union[Sequence[int], Tuple[int, int], Mapping[str, int]]


@dataclass(frozen=True)
class Span:
    """Simple representation of a character span."""

    start: int
    end: int

    def clamp(self, lower: int, upper: int) -> "Span":
        """Return a span that stays within ``[lower, upper]``."""

        new_start = min(max(self.start, lower), upper)
        new_end = min(max(self.end, lower), upper)
        if new_end < new_start:
            new_end = new_start
        return Span(new_start, new_end)


def _coerce_span(raw: SpanInput) -> Span:
    """Convert a span specification into a :class:`Span` instance."""

    if isinstance(raw, Mapping):
        start = int(raw.get("start", 0))
        end = int(raw.get("end", 0))
    else:
        if len(raw) != 2:  # type: ignore[arg-type]
            raise ValueError(f"Span sequences must have exactly two elements: {raw!r}")
        start, end = int(raw[0]), int(raw[1])  # type: ignore[index]
    if end < start:
        start, end = end, start
    return Span(start, end)


def _normalize_spans(text: str, spans: Optional[Iterable[SpanInput]]) -> List[Span]:
    """Normalize incoming span annotations.

    - Converts all inputs to :class:`Span`.
    - Clamps to the text bounds.
    - Sorts and merges overlapping spans.
    """

    if not spans:
        return []

    text_len = len(text)
    normalized: List[Span] = []
    for raw in spans:
        span = _coerce_span(raw).clamp(0, text_len)
        if span.end <= span.start:
            continue
        normalized.append(span)

    if not normalized:
        return []

    normalized.sort(key=lambda s: (s.start, s.end))
    merged: List[Span] = [normalized[0]]
    for current in normalized[1:]:
        prev = merged[-1]
        if current.start <= prev.end:
            merged[-1] = Span(prev.start, max(prev.end, current.end))
        else:
            merged.append(current)
    return merged


def _span_mask(length: int, spans: Sequence[Span]) -> List[bool]:
    """Build a boolean mask indicating highlighted character positions."""

    mask = [False] * length
    for span in spans:
        for idx in range(span.start, min(span.end, length)):
            mask[idx] = True
    return mask


def _resolve_font(font: Optional[ImageFont.ImageFont]) -> ImageFont.ImageFont:
    """Retrieve a usable font instance."""

    if font is not None:
        return font
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=14)
    except (OSError, IOError):
        return ImageFont.load_default()


def _measure_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    """Measure text width with a graceful fallback across Pillow releases."""

    try:
        return float(draw.textlength(text, font=font))
    except AttributeError:  # Pillow < 8.0 fallback
        return float(draw.textsize(text, font=font)[0])


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        raise ValueError(f"Invalid color specification: {color!r}")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return r, g, b


def _compute_line_breaks(
    text: str,
    max_width: int,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
) -> List[Tuple[int, int]]:
    """Split ``text`` into drawable lines constrained by ``max_width``."""

    lines: List[Tuple[int, int]] = []
    idx = 0
    text_len = len(text)

    while idx < text_len:
        if text[idx] == "\n":
            lines.append((idx, idx))
            idx += 1
            continue

        line_start = idx
        line_width = 0.0
        last_space = -1

        while idx < text_len:
            char = text[idx]
            if char == "\n":
                lines.append((line_start, idx))
                idx += 1
                break

            probe = "    " if char == "\t" else char
            char_width = _measure_width(draw, probe, font)

            if line_width + char_width > max_width and line_width > 0:
                if last_space >= line_start:
                    lines.append((line_start, last_space))
                    idx = last_space + 1
                else:
                    lines.append((line_start, idx))
                break

            line_width += char_width
            if char == " ":
                last_space = idx
            idx += 1
        else:
            lines.append((line_start, text_len))
            idx = text_len

    if not lines:
        lines.append((0, 0))
    return lines


def _render_text_panel(
    text: str,
    spans: Sequence[Span],
    highlight_color: str,
    title: str,
    *,
    width: int = 720,
    margin: int = 18,
    font: Optional[ImageFont.ImageFont] = None,
    title_font: Optional[ImageFont.ImageFont] = None,
    line_spacing: int = 6,
) -> Image.Image:
    """Render ``text`` into an image with highlighted spans."""

    resolved_font = _resolve_font(font)
    resolved_title_font = _resolve_font(title_font)
    dummy = Image.new("RGB", (1, 1), "white")
    draw = ImageDraw.Draw(dummy)

    max_text_width = max(1, width - 2 * margin)
    line_spans = _compute_line_breaks(text, max_text_width, draw, resolved_font)
    highlight_mask = _span_mask(len(text), spans)

    ascent, descent = resolved_font.getmetrics()
    line_height = ascent + descent + line_spacing

    title_height = 0
    if title:
        try:
            title_bbox = draw.textbbox((0, 0), title, font=resolved_title_font)
            title_raw_height = title_bbox[3] - title_bbox[1]
        except AttributeError:
            title_raw_height = draw.textsize(title, font=resolved_title_font)[1]
        title_height = title_raw_height + line_spacing

    panel_height = max(1, title_height + len(line_spans) * line_height + margin * 2)
    panel = Image.new("RGB", (width, panel_height), "white")
    panel_draw = ImageDraw.Draw(panel)

    y = margin
    if title:
        panel_draw.text((margin, y), title, fill="black", font=resolved_title_font)
        y += title_height

    highlight_rgb = _hex_to_rgb(highlight_color)

    for start, end in line_spans:
        x = margin
        for idx in range(start, end):
            char = text[idx]
            probe = "    " if char == "\t" else char
            char_width = _measure_width(panel_draw, probe, resolved_font)
            if highlight_mask[idx]:
                panel_draw.rectangle(
                    [x, y, x + char_width, y + ascent + descent],
                    fill=highlight_rgb,
                )
            panel_draw.text((x, y), probe, fill="black", font=resolved_font)
            x += char_width
        y += line_height

    return panel


def _prepare_image(image: Image.Image) -> np.ndarray:
    """Convert the input ``image`` into an array suitable for plotting."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance")
    return np.asarray(image.convert("RGB"))


def _prepare_text_panel(
    text: str,
    spans: Optional[Iterable[SpanInput]],
    *,
    highlight_color: str,
    title: str,
    panel_width: int,
) -> np.ndarray:
    normalized = _normalize_spans(text, spans)
    panel_image = _render_text_panel(text, normalized, highlight_color, title, width=panel_width)
    return np.asarray(panel_image)


def create_visualization(
    image: Image.Image,
    reference: str,
    generation: str,
    mistakes: Optional[Iterable[SpanInput]] = None,
    omissions: Optional[Iterable[SpanInput]] = None,
    *,
    figsize: Tuple[float, float] = (12.0, 8.0),
    panel_width: int = 720,
    mistake_color: str = "#f8d7da",
    omission_color: str = "#d1e7ff",
) -> plt.Figure:
    """Build a matplotlib figure highlighting mistakes and omissions.

    Parameters
    ----------
    image:
        The image as a PIL image instance.
    reference / generation:
        The reference and generated descriptions.
    mistakes:
        Iterable of spans (``start``/``end``) that should be highlighted in the
        generation text as mistakes.
    omissions:
        Iterable of spans that should be highlighted in the reference text as
        omissions.
    figsize:
        Matplotlib figure size in inches.
    panel_width:
        Width in pixels for the rendered text panels.
    mistake_color / omission_color:
        Colors (hex strings) used for highlighting mistakes (generation) and
        omissions (reference).

    Returns
    -------
    matplotlib.figure.Figure
        A figure containing the composed visualization.
    """

    image_array = _prepare_image(image)
    reference_panel = _prepare_text_panel(
        reference,
        omissions,
        highlight_color=omission_color,
        title="Reference (omissions in blue)",
        panel_width=panel_width,
    )
    generation_panel = _prepare_text_panel(
        generation,
        mistakes,
        highlight_color=mistake_color,
        title="Generation (mistakes in red)",
        panel_width=panel_width,
    )

    fig = plt.figure(figsize=figsize)
    image_width = max(1, image_array.shape[1])
    grid = gridspec.GridSpec(2, 2, width_ratios=[1.0, panel_width / image_width], figure=fig)

    image_ax = fig.add_subplot(grid[:, 0])
    image_ax.imshow(image_array)
    image_ax.axis("off")
    image_ax.set_title("Image", fontsize=14, pad=10)

    ref_ax = fig.add_subplot(grid[0, 1])
    ref_ax.imshow(reference_panel)
    ref_ax.axis("off")

    gen_ax = fig.add_subplot(grid[1, 1])
    gen_ax.imshow(generation_panel)
    gen_ax.axis("off")

    fig.tight_layout()
    return fig


def show_visualization(
    image: Image.Image,
    reference: str,
    generation: str,
    mistakes: Optional[Iterable[SpanInput]] = None,
    omissions: Optional[Iterable[SpanInput]] = None,
    **kwargs,
) -> plt.Figure:
    """Create and display the visualization with ``plt.show()``."""

    fig = create_visualization(
        image,
        reference,
        generation,
        mistakes=mistakes,
        omissions=omissions,
        **kwargs,
    )
    plt.show(block=False)
    return fig


def save_visualization(
    image: Image.Image,
    reference: str,
    generation: str,
    mistakes: Optional[Iterable[SpanInput]] = None,
    omissions: Optional[Iterable[SpanInput]] = None,
    save_path: Union[str, Path] = Path("visualization.png"),
    *,
    dpi: int = 150,
    close: bool = True,
    **kwargs,
) -> Path:
    """Create the visualization and persist it to ``save_path``."""

    fig = create_visualization(
        image,
        reference,
        generation,
        mistakes=mistakes,
        omissions=omissions,
        **kwargs,
    )
    path = Path(save_path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
