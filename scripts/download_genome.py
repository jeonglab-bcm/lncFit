"""Download the Ensembl GRCh37 (hg19) primary assembly reference FASTA to data/raw/genome/.

hg19/GRCh37 was chosen to match the lncRNA GTF (data/raw/human.lncRNA.hg19.gtf), whose
annotations come from Sarropoulos et al. 2019 (PMC6660317) and are built on hg19 (Ensembl
release 75). A prior version of this script downloaded GDC's GRCh38.d1.vd1 (hg38) — paired
with an hg19-coordinate GTF that produces genomically wrong sequences (see issue #66). hg38
support was dropped entirely rather than kept as an option, to avoid silently reintroducing
that mismatch.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import urllib.request
from pathlib import Path

_URL = (
    "https://ftp.ensembl.org/pub/grch37/release-116/fasta/homo_sapiens/dna/"
    "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
)
_FILENAME = "Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
_MD5 = "5f994d0cfb1f0c19050da6ab2d613873"
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


def extract(gz_path: Path) -> Path:
    """Decompress the .fa.gz next to it, returning the .fa path (what pyfaidx needs)."""
    fa_path = gz_path.with_suffix("")  # strip .gz
    if fa_path.exists():
        return fa_path
    print(f"Extracting {gz_path} -> {fa_path} …")
    with gzip.open(gz_path, "rb") as f_in, open(fa_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return fa_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_DEST,
        help=f"Directory to save the file (default: {_DEFAULT_DEST})",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Also decompress to a plain .fa file (required by pyfaidx.Fasta).",
    )
    args = parser.parse_args()
    gz = download(args.dest)
    if args.extract:
        extract(gz)
