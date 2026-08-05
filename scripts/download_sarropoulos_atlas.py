"""Download the Sarropoulos developmental lncRNA expression atlas, subset to our targets.

The paper's stated signature for essential lncRNAs is that they are "highly expressed early
and broadly in development". The model already carries the atlas's SUMMARY statistics, because
mmc2 S1A's `Tissue tau`, `Time tau`, `Dynamic` and `Count dynamic tissues` are all derived
from this same atlas. What it does not carry is the underlying magnitude and the DIRECTION of
the timing: tau measures how uneven a profile is, not how high it sits or whether the peak is
prenatal. That is the gap this fills.

Why the join is exact, contrary to the assumption that it would be lossy: the challenge's
targets are `Hum_XLOC_*` identifiers from data/raw/human.lncRNA.hg19.gtf, whose attributes say
`gene_source Kaessmann`. That annotation IS the Sarropoulos catalogue, so the atlas is keyed on
the very identifiers we already have. All 5,496 targets match exactly -- no coordinate join, no
GENCODE/GTEx intermediary, no dropped genes.

Source: Sarropoulos et al. 2019, Nature 571:510-514, Supplementary data 2 (HumanRPKMs.txt),
retrieved through the Europe PMC supplementary-files endpoint for PMC6660317, because both the
PMC per-file URLs and the Kaessmann Shiny app serve downloads behind session redirects.

The upstream archive is ~481 MB and HumanRPKMs.txt alone is 206 MB across 85,037 genes. Only
the 5,496 target rows are kept, written gzipped (a few MB). Intermediates are removed unless
--keep-download is passed.

Legality under the no-measured-depletion rule (docs/PARTICIPATE.md): this is baseline RNA
abundance in human developmental tissues. It is not a knockdown outcome from any cell line or
day, and it is the same category of feature -- expression -- that the rules list as legal.

Usage:
  python scripts/download_sarropoulos_atlas.py
"""
import argparse
import gzip
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
_SUPP_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6660317/supplementaryFiles"
_OUTER = "EMS83300-supplement-Supplementary_data_2.zip"
_INNER = "HumanRPKMs.txt"
_MMC2 = REPO / "data/raw/mmc2.xlsx"
_OUT = REPO / "data/external/sarropoulos_human_lncrna_rpkm.tsv.gz"

# The proliferation markers Figure S11A correlates essential lncRNAs against. Kept in a
# separate small file because they are protein-coding genes, not targets, and the main matrix
# is keyed by lncRNA. Same 297 sample columns, so the two files align positionally.
_MARKERS = {"ENSG00000132646": "PCNA", "ENSG00000148773": "MKI67"}
_OUT_MARKERS = REPO / "data/external/sarropoulos_proliferation_markers.tsv.gz"


def _targets() -> list[str]:
    s1a = pd.read_excel(_MMC2, sheet_name="S1A", header=2)
    return s1a.loc[s1a["lncRNA"].notna(), "lncRNA"].astype(str).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-download", action="store_true",
                    help="keep the 481 MB archive and the 206 MB extracted matrix")
    ap.add_argument("--work", default=None, help="scratch directory for downloads")
    args = ap.parse_args()

    work = Path(args.work) if args.work else _OUT.parent / "_sarropoulos_tmp"
    work.mkdir(parents=True, exist_ok=True)
    supp, rpkm = work / "epmc_supp.zip", work / _INNER

    if not rpkm.exists():
        if not supp.exists():
            print(f"downloading {_SUPP_URL} (~481 MB) ...", flush=True)
            subprocess.run(["curl", "-sS", "-L", "-o", str(supp), _SUPP_URL], check=True)
        print(f"extracting {_OUTER} -> {_INNER} ...", flush=True)
        with zipfile.ZipFile(supp) as z:
            z.extract(_OUTER, path=work)
        with zipfile.ZipFile(work / _OUTER) as z:
            z.extract(_INNER, path=work)
        (work / _OUTER).unlink(missing_ok=True)

    targets = _targets()
    want = set(targets)
    print(f"{len(want):,} targets; subsetting {rpkm.name} ...", flush=True)

    with open(rpkm) as fh:
        header = fh.readline().rstrip("\n")
        kept, markers = {}, {}
        for line in fh:
            gid, _, rest = line.partition("\t")
            if gid in want:
                kept[gid] = rest.rstrip("\n")
            elif gid in _MARKERS:
                markers[gid] = rest.rstrip("\n")

    missing = [t for t in targets if t not in kept]
    print(f"matched {len(kept):,}/{len(targets):,} "
          f"({100 * len(kept) / len(targets):.1f}%); {len(missing)} unmatched")
    if missing:
        print(f"  first unmatched: {missing[:5]}", file=sys.stderr)

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(_OUT, "wt") as out:
        out.write("lncRNA\t" + header + "\n")
        for t in targets:                       # preserve S1A order, skip any unmatched
            if t in kept:
                out.write(f"{t}\t{kept[t]}\n")
    print(f"-> {_OUT} ({_OUT.stat().st_size / 1e6:.1f} MB, "
          f"{len(header.split(chr(9)))} sample columns)")

    missing_markers = [g for g in _MARKERS if g not in markers]
    if missing_markers:
        print(f"  WARNING markers not found: "
              f"{[_MARKERS[g] for g in missing_markers]}", file=sys.stderr)
    with gzip.open(_OUT_MARKERS, "wt") as out:
        out.write("gene\tsymbol\t" + header + "\n")
        for gid, sym in _MARKERS.items():
            if gid in markers:
                out.write(f"{gid}\t{sym}\t{markers[gid]}\n")
    print(f"-> {_OUT_MARKERS} ({len(markers)}/{len(_MARKERS)} markers: "
          f"{', '.join(_MARKERS[g] for g in markers)})")

    if not args.keep_download:
        shutil.rmtree(work, ignore_errors=True)
        print(f"removed {work} (pass --keep-download to keep it)")


if __name__ == "__main__":
    main()
