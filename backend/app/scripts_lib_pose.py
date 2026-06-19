"""Pure helpers for polygon->pose label conversion (unit-tested)."""
from __future__ import annotations


def order_tl_tr_br_bl(
    pts: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Order 4 (x,y) points into TL, TR, BR, BL by coordinate geometry.

    Index-based selection (not object identity) so coincident corner values in
    a degenerate quad are still partitioned correctly.
    """
    order = sorted(range(len(pts)), key=lambda i: pts[i][0] + pts[i][1])
    tl, br = pts[order[0]], pts[order[-1]]
    # of the remaining two, larger (x - y) is TR, smaller is BL
    mid = [pts[order[1]], pts[order[2]]]
    mid.sort(key=lambda p: p[0] - p[1])
    bl, tr = mid[0], mid[1]
    return tl, tr, br, bl


def polygon_line_to_pose(line: str) -> str:
    """`class x1 y1 x2 y2 x3 y3 x4 y4` -> pose `class cx cy w h x y v ...`."""
    parts = line.split()
    cls = parts[0]
    coords = list(map(float, parts[1:9]))
    pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    ordered = order_tl_tr_br_bl(pts)

    xs = [p[0] for p in ordered]
    ys = [p[1] for p in ordered]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    cx = max(0.0, min(1.0, (xmin + xmax) / 2))
    cy = max(0.0, min(1.0, (ymin + ymax) / 2))
    w = max(0.0, min(1.0, xmax - xmin))
    h = max(0.0, min(1.0, ymax - ymin))

    kpts = " ".join(f"{p[0]:.6f} {p[1]:.6f} 2" for p in ordered)
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts}"
