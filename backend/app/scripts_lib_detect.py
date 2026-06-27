"""Pure helpers for polygon → detect (bbox) label conversion (unit-tested)."""
from __future__ import annotations


def polygon_line_to_detect(line: str) -> str | None:
    """`class x1 y1 x2 y2 x3 y3 x4 y4` (9 fields) → detect `class cx cy w h`.

    Also accepts 5-point polygons (11 fields: class + 5 xy pairs) by
    computing the axis-aligned bounding box over all supplied points.
    Lines with fewer than 5 fields or an odd number of coordinate fields
    are silently skipped (return None).
    """
    parts = line.split()
    if len(parts) < 5:
        return None

    cls = parts[0]
    coords = parts[1:]
    if len(coords) % 2 != 0:
        # Malformed: drop trailing odd field
        coords = coords[: len(coords) - 1]
    if len(coords) < 4:
        return None

    try:
        vals = list(map(float, coords))
    except ValueError:
        return None

    xs = vals[0::2]
    ys = vals[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    cx = max(0.0, min(1.0, (xmin + xmax) / 2))
    cy = max(0.0, min(1.0, (ymin + ymax) / 2))
    w  = max(0.0, min(1.0, xmax - xmin))
    h  = max(0.0, min(1.0, ymax - ymin))

    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
