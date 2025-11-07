"""Utilities for visualizing DOCENT annotations using HTML/CSS.

This module generates clean, interactive HTML visualizations with proper
text highlighting and layout control.

Example
-------
```python
from pathlib import Path
from PIL import Image

from visualize_html import create_visualization, save_visualization

image = Image.open(Path("example.jpg"))
reference = "A woman in a white dress holds a golden chalice."
generation = "A woman in a blue dress holds a silver cup."
mistakes = [{"start": 16, "end": 25}, {"start": 38, "end": 48}]
omissions = [{"start": 24, "end": 39}]

html = create_visualization(image, reference, generation, mistakes, omissions)
save_visualization(
    image,
    reference,
    generation,
    mistakes,
    omissions,
    Path("visualization.html"),
)
```
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import argparse
import json

from PIL import Image
from datasets import load_dataset

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
    """Normalize incoming span annotations."""

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


def _create_highlighted_html(
    text: str,
    spans: List[Span],
    highlight_color: str,
) -> str:
    """Create HTML with highlighted spans."""

    if not spans:
        # Escape HTML and preserve line breaks
        import html

        return html.escape(text).replace("\n", "<br>")

    # Sort spans by start position
    sorted_spans = sorted(spans, key=lambda s: s.start)

    # Build the HTML with highlights
    import html as html_module

    result = []
    last_end = 0

    for span in sorted_spans:
        # Add text before the span
        if span.start > last_end:
            escaped = html_module.escape(text[last_end : span.start])
            result.append(escaped.replace("\n", "<br>"))

        # Add highlighted text
        highlighted_text = html_module.escape(text[span.start : span.end])
        result.append(
            f'<mark style="background-color: {highlight_color}; padding: 2px 4px; border-radius: 3px;">{highlighted_text}</mark>'
        )

        last_end = span.end

    # Add remaining text
    if last_end < len(text):
        escaped = html_module.escape(text[last_end:])
        result.append(escaped.replace("\n", "<br>"))

    return "".join(result)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 data URL."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_visualization(
    image: Image.Image,
    reference: str,
    generation: str,
    mistakes: Optional[Iterable[SpanInput]] = None,
    omissions: Optional[Iterable[SpanInput]] = None,
    *,
    mistake_color: str = "#ffcccc",
    omission_color: str = "#cce5ff",
) -> str:
    """Build an HTML visualization highlighting mistakes and omissions.

    Parameters
    ----------
    image:
        The image as a PIL image instance.
    reference / generation:
        The reference and generated descriptions.
    mistakes:
        Iterable of spans that should be highlighted in the generation text.
    omissions:
        Iterable of spans that should be highlighted in the reference text.
    mistake_color / omission_color:
        Colors (hex strings) used for highlighting.

    Returns
    -------
    str
        HTML string containing the visualization.
    """

    # Convert image to base64
    img_data = _image_to_base64(image)

    # Normalize spans
    normalized_omissions = _normalize_spans(reference, omissions)
    normalized_mistakes = _normalize_spans(generation, mistakes)

    # Create highlighted HTML text
    reference_html = _create_highlighted_html(
        reference, normalized_omissions, omission_color
    )
    generation_html = _create_highlighted_html(
        generation, normalized_mistakes, mistake_color
    )

    # Create the HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DOCENT Visualization</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            display: grid;
            grid-template-columns: 40% 60%;
            gap: 20px;
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .image-panel {{
            display: flex;
            align-items: center;
            justify-content: center;
            grid-row: 1 / 3;
        }}
        
        .image-panel img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .text-panel {{
            background: white;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow-wrap: break-word;
            word-wrap: break-word;
        }}
        
        .text-panel h3 {{
            margin-bottom: 15px;
            color: #333;
            font-size: 18px;
            font-weight: 600;
        }}
        
        .text-content {{
            line-height: 1.8;
            color: #444;
            font-size: 15px;
        }}
        
        mark {{
            border-radius: 3px;
            padding: 2px 4px;
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 14px;
            color: #666;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-box {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #ccc;
        }}
        
        @media (max-width: 1200px) {{
            .container {{
                grid-template-columns: 1fr;
                grid-template-rows: auto auto auto;
            }}
            
            .image-panel {{
                grid-row: 1 / 2;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="image-panel">
            <img src="{img_data}" alt="Painting">
        </div>
        
        <div class="text-panel">
            <h3>Reference</h3>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box" style="background-color: {omission_color};"></div>
                    <span>Omissions</span>
                </div>
            </div>
            <div class="text-content">
                {reference_html}
            </div>
        </div>
        
        <div class="text-panel">
            <h3>Generation</h3>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box" style="background-color: {mistake_color};"></div>
                    <span>Mistakes</span>
                </div>
            </div>
            <div class="text-content">
                {generation_html}
            </div>
        </div>
    </div>
</body>
</html>
"""

    return html


def save_visualization(
    image: Image.Image,
    reference: str,
    generation: str,
    mistakes: Optional[Iterable[SpanInput]] = None,
    omissions: Optional[Iterable[SpanInput]] = None,
    save_path: Union[str, Path] = Path("visualization.html"),
    **kwargs,
) -> Path:
    """Create the visualization and save it as an HTML file."""

    html = create_visualization(
        image,
        reference,
        generation,
        mistakes=mistakes,
        omissions=omissions,
        **kwargs,
    )

    path = Path(save_path)
    path.write_text(html, encoding="utf-8")

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize a DOCENT granular annotation as HTML")
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--save-path", type=str, default="visualization.html")
    args = parser.parse_args()

    dataset = load_dataset("amitha/docent-eval-granular")

    assert args.idx < len(dataset["test"]), f"Index {args.idx} out of range!"

    item = dataset["test"][args.idx]
    save_visualization(
        item["image"],
        item["reference"],
        item["generation"],
        json.loads(item["mistakes"]),
        json.loads(item["omissions"]),
        Path(args.save_path),
    )

    print(f"Visualization saved to {args.save_path}")
