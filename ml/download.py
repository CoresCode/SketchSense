"""Download Google QuickDraw sketches for our class list.

Deliberate choice: we stream the *simplified NDJSON* instead of the 28x28
`numpy_bitmap` mirror.  A bitmap file is ~92 MB per class, while one class of
NDJSON only needs ~10 MB for 12k sketches -- and stopping the stream early is
legal because NDJSON is line-delimited.  We also get the original strokes, so we
can rasterize at 64x64 instead of being stuck at 28x28.

Files land in ml/data/raw/<class>.ndjson and every class is resumable: re-running
the script skips anything already complete.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Windows consoles default to cp1252 and choke on any non-ASCII class name we echo
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
USER_AGENT = "SketchSense/0.1 (student portfolio project; contact: corescode.com)"


def read_classes(path: Path) -> list[str]:
    out = []
    # utf-8-sig: PowerShell's Set-Content writes a BOM, which would otherwise
    # become part of the first class name and 404 the request.
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def slugify(word: str) -> str:
    return word.replace(" ", "_").replace("-", "_").lower()


def count_lines(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def fetch_one(
    word: str, max_sketches: int, out_dir: Path, retries: int = 5
) -> tuple[str, int, bool, str | None]:
    """Returns (word, sketches, downloaded_now, error). Never raises: one flaky class must
    not abort an 84-class batch."""
    dest = out_dir / f"{slugify(word)}.ndjson"
    if dest.exists():
        have = count_lines(dest)
        if have >= max_sketches:
            return word, have, False, None

    url = BASE_URL.format(urllib.parse.quote(word))
    tmp = dest.with_suffix(".part")
    last_error = "unknown error"

    for attempt in range(1, retries + 1):
        written = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=90) as resp, tmp.open(
                "w", encoding="utf-8", newline="\n"
            ) as f:
                for raw in resp:
                    f.write(raw.decode("utf-8"))
                    written += 1
                    if written >= max_sketches:
                        break
            if written == 0:
                raise RuntimeError("empty response")
            tmp.replace(dest)
            return word, written, True, None
        except Exception as exc:  # noqa: BLE001 - every network error is worth a retry
            last_error = f"{type(exc).__name__}: {exc}"
            waited = min(5 * attempt, 20)
            print(f"  [{word}] attempt {attempt}/{retries} failed: {exc} (retry in {waited}s)", flush=True)
            time.sleep(waited)

    return word, 0, False, last_error


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.txt"))
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "raw")
    ap.add_argument("--max-sketches", type=int, default=12000, help="sketches per class to keep")
    ap.add_argument("--workers", type=int, default=2, help="keep low on a phone hotspot")
    ap.add_argument("--first", type=int, default=0, help="only do the first N classes (smoke test)")
    ap.add_argument("--retry-failed", action="store_true", help="ignore existing files, re-fetch")
    args = ap.parse_args()

    words = read_classes(args.classes)
    if args.first:
        words = words[: args.first]
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{len(words)} classes -> {args.out} ({args.max_sketches} sketches each)", flush=True)
    started = time.time()
    done = 0
    failures: list[tuple[str, str]] = []

    def report(word: str, n: int, fresh: bool, error: str | None) -> None:
        nonlocal done
        done += 1
        elapsed = time.time() - started
        per = elapsed / done
        eta = per * (len(words) - done)
        if error:
            failures.append((word, error))
            print(f"[{done}/{len(words)}] {word:<18} FAILED: {error}", flush=True)
            return
        state = "downloaded" if fresh else "cached"
        print(
            f"[{done}/{len(words)}] {word:<18} {n:>6} sketches ({state})"
            f"  elapsed {elapsed/60:5.1f}m  eta {eta/60:5.1f}m",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(fetch_one, w, args.max_sketches, args.out) for w in words]
        for fut in futures:
            report(*fut.result())

    # Second, gentler pass: a hotspot that dropped four connections in a row often
    # recovers if we stop hammering it with parallel sockets.
    if failures:
        print(f"\nretrying {len(failures)} failed class(es) one at a time …", flush=True)
        retry = failures
        failures = []
        done = 0
        for word, _ in retry:
            report(*fetch_one(word, args.max_sketches, args.out, retries=6))

    total_mb = sum(p.stat().st_size for p in args.out.glob("*.ndjson")) / 1e6
    ok = len(words) - len(failures)
    print(f"\n{ok}/{len(words)} classes, {total_mb:.0f} MB in {args.out} over {(time.time()-started)/60:.1f} min")
    if failures:
        print("still missing:", ", ".join(w for w, _ in failures))
        print("re-run the same command to resume; completed classes are skipped")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
