"""Download the GDC GRCh38.d1.vd1 reference FASTA to data/raw/genome/."""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

_URL = "https://api.gdc.cancer.gov/data/254f697d-310d-4d7d-a27b-27fbf767a834"
_FILENAME = "GRCh38.d1.vd1.fa.tar.gz"
_MD5 = "3ffbcfe2d05d43206f57f81ebb251dc9"
_DEFAULT_DEST = Path("data/raw/genome")


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / _FILENAME

    if out.exists():
        print(f"Already exists: {out} — verifying checksum …")
        if _md5(out) == _MD5:
            print("Checksum OK, skipping download.")
            return out
        print("Checksum mismatch — re-downloading.")
        out.unlink()

    print(f"Downloading {_FILENAME} to {out} …")

    def _progress(count: int, block: int, total: int) -> None:
        pct = min(count * block / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(_URL, out, reporthook=_progress)
    print()

    digest = _md5(out)
    if digest != _MD5:
        out.unlink()
        raise RuntimeError(f"Checksum mismatch: expected {_MD5}, got {digest}")

    print(f"Checksum OK. Saved to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_DEST,
        help=f"Directory to save the file (default: {_DEFAULT_DEST})",
    )
    args = parser.parse_args()
    download(args.dest)
