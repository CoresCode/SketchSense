"""Crop the hero screenshot into a 1200x630 social preview card.

Open Graph images are the first thing anyone sees when the link is pasted into Slack, Discord
or LinkedIn, so this is part of the docs build rather than a one-off. Takes the 2x hero shot
from tools/capture.mjs and cuts the band that holds the brand, the headline, the mode tabs and
the top of the canvas.

    python ml/og.py            # docs/screenshot.png -> web/public/og.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
REPO = HERE.parent
W, H = 1200, 630


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=REPO / "docs" / "screenshot.png")
    ap.add_argument("--out", type=Path, default=REPO / "web" / "public" / "og.png")
    ap.add_argument("--top", type=float, default=0.0, help="fraction of source height to start at")
    args = ap.parse_args()

    img = Image.open(args.src).convert("RGB")
    # Crop the widest band matching the card's aspect ratio, starting at --top.
    band_h = round(img.width * H / W)
    top = round(img.height * args.top)
    top = max(0, min(top, img.height - band_h))
    card = img.crop((0, top, img.width, top + band_h)).resize((W, H), Image.LANCZOS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    card.save(args.out, optimize=True)
    print(f"{args.out}  {W}x{H}  {args.out.stat().st_size / 1024:.0f} KB from {args.src.name}")


if __name__ == "__main__":
    main()
