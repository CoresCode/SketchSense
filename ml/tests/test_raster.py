"""Rasteriser contract tests. No pytest needed: run it directly.

    ..\\.venv\\Scripts\\python.exe tests\\test_raster.py

These are the invariants the browser implementation (web/src/lib/raster.ts) also relies on,
so a change here without a matching change there is a bug, not a tuning decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raster import CANVAS, PAD, normalise_strokes, rasterize, rasterize_uint8  # noqa: E402


def circle(cx: float, cy: float, r: float, n: int = 40) -> list[list[float]]:
    t = np.linspace(0, 2 * np.pi, n)
    return [(cx + r * np.cos(t)).tolist(), (cy + r * np.sin(t)).tolist()]


def test_shape_and_range() -> None:
    img = rasterize([circle(128, 128, 60)])
    assert img.shape == (64, 64), img.shape
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert img.max() == 1.0, "a drawn stroke should reach full intensity"


def test_empty_and_degenerate() -> None:
    assert rasterize([]).sum() == 0.0, "no strokes must mean no ink"
    blank = np.zeros((64, 64), dtype=np.uint8)
    assert np.array_equal(rasterize_uint8([]), blank)

    # A tap is a single point. Both rasterisers draw it as a dot near the centre, because
    # in the browser a tap is how people draw eyes and punctuation.
    dot = rasterize([[[10.0], [10.0]]])
    assert dot.sum() > 0, "a single-point stroke should still leave a dot"
    ys, xs = np.nonzero(dot)
    assert abs(xs.mean() - 32) < 1.5 and abs(ys.mean() - 32) < 1.5, "dot should be centred"


def test_bbox_normalisation_is_translation_invariant() -> None:
    """A drawing made in the corner must rasterise like the same drawing in the middle."""
    a = rasterize([circle(128, 128, 50)])
    b = rasterize([circle(40, 200, 50)])
    c = rasterize([circle(230, 30, 50)])
    assert np.abs(a - b).mean() < 0.02
    assert np.abs(a - c).mean() < 0.02


def test_scale_invariance() -> None:
    """A small drawing and a large one normalise to the same tensor."""
    small = rasterize([circle(128, 128, 15)])
    large = rasterize([circle(128, 128, 120)])
    assert np.abs(small - large).mean() < 0.03


def test_normalise_centres_and_pads() -> None:
    strokes = [[[10.0, 20.0], [30.0, 40.0]]]
    lines = normalise_strokes(strokes, canvas=CANVAS, pad=PAD)
    assert len(lines) == 1
    xs = lines[0][:, 0]
    ys = lines[0][:, 1]
    # longest side spans canvas - 2*pad, centred
    assert abs((xs.max() - xs.min()) - (CANVAS - 2 * PAD)) < 1e-3
    assert abs((ys.max() - ys.min()) - (CANVAS - 2 * PAD)) < 1e-3
    assert abs((xs.max() + xs.min()) / 2 - CANVAS / 2) < 1e-3
    assert abs((ys.max() + ys.min()) / 2 - CANVAS / 2) < 1e-3


def _dilate(mask: np.ndarray, k: int = 2) -> np.ndarray:
    out = mask.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def test_ink_is_stable_across_point_density() -> None:
    """Pointer sampling rate changes how many points a stroke has, not what it looks like.

    Compared with a 2px tolerance: thin antialiased lines make raw IoU meaningless, but
    "every bit of ink has a neighbour within 2px" is exactly the property we care about.
    """
    coarse = rasterize([circle(128, 128, 60, n=24)]) > 0.3
    fine = rasterize([circle(128, 128, 60, n=200)]) > 0.3
    cover_coarse = (coarse & _dilate(fine)).sum() / max(coarse.sum(), 1)
    cover_fine = (fine & _dilate(coarse)).sum() / max(fine.sum(), 1)
    assert min(cover_coarse, cover_fine) > 0.97, f"coverage {cover_coarse:.3f} / {cover_fine:.3f}"


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
