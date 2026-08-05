"""Download UCSC phastCons conserved elements, subset to our lncRNA exons.

Conservation is not one of the paper's analyses. Its only evolutionary measure is S1A's `Age`,
a 7-category estimate of when the gene first appeared, tested by Fisher's exact in Figure 2A.
This is our own idea, and it should be described that way.

The motivation is that `Age` is the only block whose contribution keeps the same sign on all
three folds (scripts/sweep_age_ablation.py: +0.0058 / +0.0034 / +0.0001), which no other block
managed. Age and conservation answer related but different questions. Age asks whether a
recognisable copy of the gene exists in other species at all -- one coarse label for the whole
gene, and "unknown" for 384 of our targets. Conservation asks how hard selection has held the
sequence in place, base by base. A gene can be old but freely drifting, or recent but tightly
constrained; `Age` cannot tell those apart.

Why elements rather than per-base scores: pyBigWig cannot be built on this box (no Python.h,
and UCSC ships no linux.aarch64 binaries), so the 9.6 GB phyloP and 5.8 GB phastCons bigWigs
are unreadable here. phastConsElements is the same underlying model's output in plain text --
UCSC's called conserved elements with lod scores -- and needs no compiler. It is coarser than
per-base phyloP, which is a real limitation and is why a null result here should not be read as
closing out conservation entirely.

hg19 throughout, matching data/raw/human.lncRNA.hg19.gtf, so no liftover. The GTF names
chromosomes Ensembl-style (1, 2, X); UCSC uses chr1, chr2, chrX, and the mapping is applied
here.

The full table is ~88 MB and mostly irrelevant to us, so only elements overlapping a target's
exons are kept.

Usage:
  python scripts/download_phastcons_elements.py
"""
import argparse
import gzip
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_URL = ("https://hgdownload.soe.ucsc.edu/goldenPath/hg19/database/"
        "phastConsElements100way.txt.gz")
_GTF = REPO / "data/raw/human.lncRNA.hg19.gtf"
_OUT = REPO / "data/external/phastcons_elements_hg19_100way.tsv.gz"


def load_exons(gtf: Path) -> dict[str, list[tuple[int, int]]]:
    """chrom (UCSC-named) -> merged exon intervals, over every lncRNA in the GTF."""
    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            # GTF is 1-based inclusive; convert to 0-based half-open to match UCSC.
            by_chrom[f"chr{f[0]}"].append((int(f[3]) - 1, int(f[4])))
    merged = {}
    for c, iv in by_chrom.items():
        iv.sort()
        out = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        merged[c] = [(s, e) for s, e in out]
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None)
    ap.add_argument("--keep-download", action="store_true")
    args = ap.parse_args()

    work = Path(args.work) if args.work else _OUT.parent / "_phastcons_tmp"
    work.mkdir(parents=True, exist_ok=True)
    raw = work / "phastConsElements100way.txt.gz"
    if not raw.exists():
        print(f"downloading {_URL} (~88 MB) ...", flush=True)
        subprocess.run(["curl", "-sS", "-L", "-o", str(raw), _URL], check=True)

    exons = load_exons(_GTF)
    total_bp = sum(e - s for iv in exons.values() for s, e in iv)
    print(f"{len(exons)} chromosomes, {sum(len(v) for v in exons.values()):,} merged exon "
          f"intervals, {total_bp:,} exonic bp")

    kept = 0
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "rt") as fh, gzip.open(_OUT, "wt") as out:
        out.write("chrom\tstart\tend\tlod\tscore\n")
        # UCSC table: bin, chrom, chromStart, chromEnd, name ("lod=NNN"), score
        ptr = {c: 0 for c in exons}
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 6:
                continue
            chrom, start, end = f[1], int(f[2]), int(f[3])
            iv = exons.get(chrom)
            if not iv:
                continue
            # The table is sorted by (chrom, start), so a per-chromosome pointer is enough.
            i = ptr[chrom]
            while i < len(iv) and iv[i][1] <= start:
                i += 1
            ptr[chrom] = i
            if i < len(iv) and iv[i][0] < end:
                lod = f[4].split("=")[-1] if "=" in f[4] else ""
                out.write(f"{chrom}\t{start}\t{end}\t{lod}\t{f[5]}\n")
                kept += 1
    print(f"kept {kept:,} elements overlapping lncRNA exons -> {_OUT} "
          f"({_OUT.stat().st_size / 1e6:.1f} MB)")

    if not args.keep_download:
        raw.unlink(missing_ok=True)
        try:
            work.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
