"""Download a GENCODE v36 gene_name -> gene_id map, so neighbour genes can be looked up.

mmc2 S1A names each lncRNA's closest protein-coding gene by SYMBOL ("AGRN", "AL669831.1"),
while the expression sheets S1D/S1F are keyed by Ensembl gene ID. Nothing in the repo bridges
the two, which blocks the one neighbour feature likely to matter: how strongly a lncRNA's
protein-coding neighbour is expressed in each cell line. Distance and the neighbour's DepMap
essentiality were already testable and both came out flat; neighbour expression is the
version that varies by cell line as well as by gene, the property that made the lncRNA's own
TPM useful.

Version matters. S1D/S1F are quantified against GENCODE v36, so v36 is downloaded rather
than "current" -- symbols are renamed between releases, and a mismatch would silently drop
genes instead of erroring.

Only the gene-level records are kept (about 60k rows out of ~2.9M lines), written as a small
CSV. The 40 MB source archive is removed afterwards unless --keep-gtf is passed.

Usage:
  uv run python scripts/download_gencode_gene_map.py
"""
from __future__ import annotations

import argparse
import csv
import gzip
import re
import urllib.request
from pathlib import Path

_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/"
        "gencode.v36.annotation.gtf.gz")
_DEFAULT_GTF = Path("data/raw/gencode/gencode.v36.annotation.gtf.gz")
_DEFAULT_OUT = Path("data/external/gencode_v36_gene_map.csv")

_GENE_ID = re.compile(r'gene_id "([^"]+)"')
_GENE_NAME = re.compile(r'gene_name "([^"]+)"')
_GENE_TYPE = re.compile(r'gene_type "([^"]+)"')


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"Downloading {url}\n  -> {dest} (~40 MB) ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  done ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def parse_genes(gtf_gz: Path) -> list[dict]:
    """Extract one row per gene record: id, symbol, type."""
    rows: list[dict] = []
    with gzip.open(gtf_gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 9)
            if len(parts) < 9 or parts[2] != "gene":
                continue
            attrs = parts[8]
            gid = _GENE_ID.search(attrs)
            gname = _GENE_NAME.search(attrs)
            if not gid or not gname:
                continue
            gtype = _GENE_TYPE.search(attrs)
            rows.append({
                "gene_id": gid.group(1),
                "gene_id_base": gid.group(1).split(".")[0],
                "gene_name": gname.group(1),
                "gene_type": gtype.group(1) if gtype else "",
                "chrom": parts[0],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=_URL)
    parser.add_argument("--gtf", type=Path, default=_DEFAULT_GTF)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--keep-gtf", action="store_true",
                        help="Keep the 40 MB archive instead of deleting it after parsing.")
    args = parser.parse_args()

    gtf = download(args.url, args.gtf)
    print("Parsing gene records ...")
    rows = parse_genes(gtf)
    print(f"  {len(rows):,} genes")

    n_pc = sum(1 for r in rows if r["gene_type"] == "protein_coding")
    dupes = len(rows) - len({r["gene_name"] for r in rows})
    print(f"  {n_pc:,} protein-coding; {dupes:,} symbol(s) shared by more than one gene")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["gene_id", "gene_id_base", "gene_name", "gene_type", "chrom"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_gtf:
        gtf.unlink()
        print(f"Removed {gtf} (pass --keep-gtf to retain it)")


if __name__ == "__main__":
    main()
