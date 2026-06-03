"""Generate data/mailgraph.ico — the Windows app icon for the "Mail Graph"
launcher shortcut. Pure stdlib (no Pillow): we rasterize a clay rounded square
with a paper-white envelope glyph and wrap the PNGs in a multi-size .ico.

Run via the project venv: `python scripts/make_icon.py`. Idempotent — safe to
re-run; it just overwrites data/mailgraph.ico.
"""
from __future__ import annotations

import math
import pathlib
import struct
import zlib

# Palette mirrors the app theme (graph_app.py :root): clay on warm paper.
CLAY = (204, 120, 92, 255)    # #CC785C
PAPER = (250, 249, 245, 255)  # #FAF9F5
CLEAR = (0, 0, 0, 0)          # transparent (outside the rounded square)


def _dist_seg(px: float, py: float, ax: float, ay: float,
              bx: float, by: float) -> float:
    """Distance from point P to segment AB — used to stroke the envelope flap."""
    dx, dy = bx - ax, by - ay
    d2 = dx * dx + dy * dy
    if d2 == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / d2
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _render(n: int) -> bytes:
    """Rasterize the icon at n×n and return raw PNG scanlines (filter byte +
    RGBA per row)."""
    r = n * 0.20                       # corner radius
    ex0, ex1 = n * 0.22, n * 0.78      # envelope body x-bounds
    ey0, ey1 = n * 0.34, n * 0.66      # envelope body y-bounds
    cx = n * 0.5                        # flap apex x (centre)
    apex_y = ey0 + (ey1 - ey0) * 0.46   # flap apex y
    stroke = max(2.0, n * 0.028)        # flap line thickness
    rows = bytearray()
    for y in range(n):
        rows.append(0)                  # PNG filter type 0 (None) per scanline
        for x in range(n):
            # Rounded-square mask: clamp to the inner rect; in a corner zone,
            # drop pixels outside the corner circle to transparent.
            nx = min(max(x, r), n - 1 - r)
            ny = min(max(y, r), n - 1 - r)
            in_corner = (x < r or x > n - 1 - r) and (y < r or y > n - 1 - r)
            if in_corner and (x - nx) ** 2 + (y - ny) ** 2 > r * r:
                rows += bytes(CLEAR)
                continue
            col = CLAY
            if ex0 <= x <= ex1 and ey0 <= y <= ey1:
                col = PAPER
                # Clay flap fold: two lines from the top corners to the apex.
                if (_dist_seg(x, y, ex0, ey0, cx, apex_y) <= stroke
                        or _dist_seg(x, y, ex1, ey0, cx, apex_y) <= stroke):
                    col = CLAY
            rows += bytes(col)
    return bytes(rows)


def _chunk(typ: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))


def _png(n: int, raw: bytes) -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", n, n, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (sig + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw, 9))
            + _chunk(b"IEND", b""))


def _ico(images: list[tuple[int, bytes]]) -> bytes:
    """Wrap PNG-encoded images (Vista+ PNG-in-ICO) into a single .ico blob."""
    count = len(images)
    out = struct.pack("<HHH", 0, 1, count)            # ICONDIR
    offset = 6 + 16 * count
    blob = b""
    for size, png in images:
        dim = 0 if size >= 256 else size              # 0 means 256 in ICO
        out += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32,
                           len(png), offset)           # ICONDIRENTRY
        blob += png
        offset += len(png)
    return out + blob


def main() -> int:
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "mailgraph.ico"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(_ico([(s, _png(s, _render(s))) for s in (256, 64, 32)]))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
