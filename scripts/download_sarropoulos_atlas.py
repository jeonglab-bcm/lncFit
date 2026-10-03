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
downloaded from the article's own supplementary link on nature.com (doi
10.1038/s41586-019-1341-x, file MOESM5). The PMC copy sits behind a browser check and the
Europe PMC bundle endpoint is unreliable (503s, no resume), so neither is used by default.

The upstream zip is ~432 MB and HumanRPKMs.txt alone is 206 MB across 85,037 genes. Only
the 5,496 target rows are kept, written gzipped (a few MB). Intermediates are removed unless
--keep-download is passed.

Legality under the no-measured-depletion rule (docs/PARTICIPATE.md): this is baseline RNA
abundance in human developmental tissues. It is not a knockdown outcome from any cell line or
day, and it is the same category of feature -- expression -- that the rules list as legal.

With --all-lncrnas, every lncRNA in data/raw/human.lncRNA.hg19.gtf (31,678, not just the
5,496 screened) is kept, written to sarropoulos_all_lncrna_rpkm.tsv.gz instead. That file is
~tens of MB, regenerable, and git-ignored; it feeds self-supervised pretraining.

To reuse a copy of Supplementary Data 2 you already have (from nature.com or PMC), pass it
with --from-zip.

Usage:
  python scripts/download_sarropoulos_atlas.py
  python scripts/download_sarropoulos_atlas.py --all-lncrnas
  python scripts/download_sarropoulos_atlas.py --all-lncrnas \
      --from-zip ~/Downloads/EMS83300-supplement-Supplementary_data_2.zip
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
_SUPP_URL = ("https://media.springernature.com/original/springer-static/esm/"
             "art%3A10.1038%2Fs41586-019-1341-x/MediaObjects/41586_2019_1341_MOESM5_ESM.zip")
_INNER = "HumanRPKMs.txt"
_MMC2 = REPO / "data/raw/mmc2.xlsx"
_OUT = REPO / "data/external/sarropoulos_human_lncrna_rpkm.tsv.gz"
_OUT_ALL = REPO / "data/external/sarropoulos_all_lncrna_rpkm.tsv.gz"
_GTF = REPO / "data/raw/human.lncRNA.hg19.gtf"

# The proliferation markers Figure S11A correlates essential lncRNAs against. Kept in a
# separate small file because they are protein-coding genes, not targets, and the main matrix
# is keyed by lncRNA. Same 297 sample columns, so the two files align positionally.
_MARKERS = {"ENSG00000132646": "PCNA", "ENSG00000148773": "MKI67"}
_OUT_MARKERS = REPO / "data/external/sarropoulos_proliferation_markers.tsv.gz"


def _targets() -> list[str]:
    s1a = pd.read_excel(_MMC2, sheet_name="S1A", header=2)
    return s1a.loc[s1a["lncRNA"].notna(), "lncRNA"].astype(str).tolist()


def _all_lncrnas() -> list[str]:
    genes: dict[str, None] = {}
    with open(_GTF) as fh:
        for line in fh:
            i = line.find("gene_id ")
            if i >= 0:
                genes[line[i + 8:].split(";", 1)[0].strip().strip('"')] = None
    return list(genes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep-download", action="store_true",
                    help="keep the 432 MB archive and the 206 MB extracted matrix")
    ap.add_argument("--work", default=None, help="scratch directory for downloads")
    ap.add_argument("--all-lncrnas", action="store_true",
                    help="keep every lncRNA in the GTF, not just the screened targets")
    ap.add_argument("--from-zip", default=None,
                    help="an existing copy of Supplementary Data 2 (zip)")
    args = ap.parse_args()
    out_path = _OUT_ALL if args.all_lncrnas else _OUT

    work = Path(args.work) if args.work else _OUT.parent / "_sarropoulos_tmp"
    work.mkdir(parents=True, exist_ok=True)
    supp, rpkm = work / "supplementary_data_2.zip", work / _INNER

    if not rpkm.exists():
        zpath = Path(args.from_zip).expanduser() if args.from_zip else supp
        if not zpath.exists():
            print(f"downloading {_SUPP_URL} (~432 MB) ...", flush=True)
            # Download to .part and rename on success, so an interrupted run is never
            # mistaken for a complete archive.
            part = supp.with_suffix(".zip.part")
            subprocess.run(["curl", "-sS", "-L", "--retry", "5", "-o", str(part), _SUPP_URL],
                           check=True)
            part.rename(supp)
            zpath = supp
        print(f"extracting {_INNER} from {zpath.name} ...", flush=True)
        with zipfile.ZipFile(zpath) as z:
            member = next(n for n in z.namelist() if n.endswith(_INNER))
            with z.open(member) as src, open(rpkm, "wb") as dst:
                shutil.copyfileobj(src, dst)

    targets = _all_lncrnas() if args.all_lncrnas else _targets()
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as out:
        out.write("lncRNA\t" + header + "\n")
        for t in targets:                       # preserve S1A order, skip any unmatched
            if t in kept:
                out.write(f"{t}\t{kept[t]}\n")
    print(f"-> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(header.split(chr(9)))} sample columns)")

    if args.all_lncrnas:  # marker file is unchanged by this mode
        if not args.keep_download:
            shutil.rmtree(work, ignore_errors=True)
        return

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
