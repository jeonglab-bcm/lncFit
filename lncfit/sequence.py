"""Extract body sequences for each lncRNA gene from a GTF + FASTA.

Genome build is hg19/GRCh37, matching the lncRNA GTF's source (Sarropoulos et al.
2019, PMC6660317). A prior version of this module defaulted to an hg38 FASTA
(GDC GRCh38.d1.vd1) paired with this same hg19-coordinate GTF — a genome-build
mismatch that silently produces wrong sequences (see issue #66). hg38 support
was removed rather than kept as an option, to avoid reintroducing that mismatch.
"""
from __future__ import annotations

import gzip
import re
import shutil
from collections import defaultdict
from pathlib import Path

from pyfaidx import Fasta

_GENE_ID_RE = re.compile(r'gene_id\s+(\S+?);')
_TRANSCRIPT_ID_RE = re.compile(r'transcript_id\s+(\S+?);')
_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")

_DEFAULT_GTF = Path("data/raw/human.lncRNA.hg19.gtf")
_DEFAULT_FASTA_GZ = Path("data/raw/genome/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz")
_DEFAULT_FASTA = Path("data/raw/genome/Homo_sapiens.GRCh37.dna.primary_assembly.fa")


def _revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _ensure_fasta(fasta_gz: Path, fasta: Path) -> Path:
    """Decompress the plain-gzipped Ensembl FASTA if not already extracted."""
    if fasta.exists():
        return fasta
    if not fasta_gz.exists():
        raise FileNotFoundError(
            f"FASTA not found at {fasta} and .gz not found at {fasta_gz}. "
            "Run: uv run python scripts/download_genome.py"
        )
    print(f"Extracting {fasta_gz} → {fasta} (this may take a few minutes) …")
    with gzip.open(fasta_gz, "rb") as f_in, open(fasta, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    print("Extraction complete.")
    return fasta


def parse_gtf(gtf_path: Path | str = _DEFAULT_GTF) -> dict[str, tuple[str, int, int, str]]:
    """Parse GTF and return gene boundaries.

    Returns:
        {gene_id: (chrom, start_1based, end_1based, strand)}
        where start/end span the union of all exons across all transcripts.
    """
    bounds: dict[str, list] = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom, start, end, strand, attrs = parts[0], int(parts[3]), int(parts[4]), parts[6], parts[8]
            m = _GENE_ID_RE.search(attrs)
            if not m:
                continue
            gene_id = m.group(1)
            if gene_id not in bounds:
                bounds[gene_id] = [chrom, start, end, strand]
            else:
                bounds[gene_id][1] = min(bounds[gene_id][1], start)
                bounds[gene_id][2] = max(bounds[gene_id][2], end)
    return {gid: (v[0], v[1], v[2], v[3]) for gid, v in bounds.items()}


def extract_body_sequences(
    gene_bounds: dict[str, tuple[str, int, int, str]],
    fasta_path: Path | str = _DEFAULT_FASTA,
    fasta_gz: Path | str = _DEFAULT_FASTA_GZ,
    window: int = 1000,
) -> dict[str, tuple[str, str]]:
    """Extract first and last `window` bp of each gene's genomic span.

    GTF chromosomes are bare numbers (e.g. "1"); FASTA uses "chr" prefix —
    this function handles the mapping automatically.

    Returns:
        {gene_id: (first_window_seq, last_window_seq)}
        Sequences are on the sense strand (reverse-complemented for minus-strand genes).
        Windows are trimmed to chromosome boundaries when the gene is near a contig edge.
    """
    fasta_path = _ensure_fasta(Path(fasta_gz), Path(fasta_path))

    print(f"Loading FASTA index from {fasta_path} …")
    fa = Fasta(str(fasta_path), sequence_always_upper=True, as_raw=True)

    # Build a lookup that handles both "1" and "chr1" style names
    fasta_chroms = set(fa.keys())

    def _resolve_chrom(chrom: str) -> str | None:
        if chrom in fasta_chroms:
            return chrom
        with_prefix = f"chr{chrom}"
        if with_prefix in fasta_chroms:
            return with_prefix
        return None

    result: dict[str, tuple[str, str]] = {}
    missing_chroms: set[str] = set()

    for gene_id, (chrom, g_start, g_end, strand) in gene_bounds.items():
        resolved = _resolve_chrom(chrom)
        if resolved is None:
            missing_chroms.add(chrom)
            continue

        chrom_len = len(fa[resolved])

        # Convert GTF 1-based inclusive coords to 0-based half-open
        s = g_start - 1  # 0-based start
        e = g_end        # 0-based exclusive end

        if strand == "-":
            # 5' end of the RNA is at genomic e; 3' end is at genomic s
            first_s = max(0, e - window)
            first_e = min(chrom_len, e)
            last_s = max(0, s)
            last_e = min(chrom_len, s + window)
            first_seq = _revcomp(str(fa[resolved][first_s:first_e]))
            last_seq = _revcomp(str(fa[resolved][last_s:last_e]))
        else:
            # + strand (or unknown): 5' end is at genomic s
            first_s = max(0, s)
            first_e = min(chrom_len, s + window)
            last_s = max(0, e - window)
            last_e = min(chrom_len, e)
            first_seq = str(fa[resolved][first_s:first_e])
            last_seq = str(fa[resolved][last_s:last_e])

        result[gene_id] = (first_seq, last_seq)

    if missing_chroms:
        print(f"Warning: {len(missing_chroms)} chromosome(s) not found in FASTA and were skipped: {sorted(missing_chroms)[:10]}")

    return result


def load_body_sequences(
    gtf_path: Path | str = _DEFAULT_GTF,
    fasta_path: Path | str = _DEFAULT_FASTA,
    fasta_gz: Path | str = _DEFAULT_FASTA_GZ,
    window: int = 1000,
) -> dict[str, tuple[str, str]]:
    """Parse GTF and extract body windows in one call.

    Returns:
        {gene_id: (first_window_seq, last_window_seq)}
    """
    gene_bounds = parse_gtf(gtf_path)
    print(f"Parsed {len(gene_bounds)} genes from GTF.")
    return extract_body_sequences(gene_bounds, fasta_path, fasta_gz, window)


def extract_full_genomic_sequences(
    gene_bounds: dict[str, tuple[str, int, int, str]],
    fasta_path: Path | str = _DEFAULT_FASTA,
    fasta_gz: Path | str = _DEFAULT_FASTA_GZ,
) -> dict[str, tuple[str, str]]:
    """Extract the full genomic span for each gene (no window truncation).

    Returns:
        {gene_id: (full_seq, "")} — full span on the sense strand; second element
        is empty so the return type is compatible with body_sequences consumers.
    """
    fasta_path = _ensure_fasta(Path(fasta_gz), Path(fasta_path))

    print(f"Loading FASTA index from {fasta_path} …")
    fa = Fasta(str(fasta_path), sequence_always_upper=True, as_raw=True)
    fasta_chroms = set(fa.keys())

    def _resolve_chrom(chrom: str) -> str | None:
        if chrom in fasta_chroms:
            return chrom
        with_prefix = f"chr{chrom}"
        if with_prefix in fasta_chroms:
            return with_prefix
        return None

    result: dict[str, tuple[str, str]] = {}
    missing_chroms: set[str] = set()

    for gene_id, (chrom, g_start, g_end, strand) in gene_bounds.items():
        resolved = _resolve_chrom(chrom)
        if resolved is None:
            missing_chroms.add(chrom)
            continue
        s = g_start - 1  # 0-based
        e = g_end
        seq = str(fa[resolved][s:e])
        if strand == "-":
            seq = _revcomp(seq)
        result[gene_id] = (seq, "")

    if missing_chroms:
        print(f"Warning: {len(missing_chroms)} chromosome(s) not found: {sorted(missing_chroms)[:10]}")

    return result


def extract_spliced_sequences(
    gtf_path: Path | str = _DEFAULT_GTF,
    fasta_path: Path | str = _DEFAULT_FASTA,
    fasta_gz: Path | str = _DEFAULT_FASTA_GZ,
) -> dict[str, tuple[str, str]]:
    """Extract spliced transcript sequence for each gene using the longest transcript.

    Concatenates exon sequences in RNA order (5'→3'), reverse-complementing for
    minus-strand genes. Returns:
        {gene_id: (spliced_seq, "")} — second element empty for type compatibility.
    """
    # Parse GTF: collect exons per (gene_id, transcript_id)
    gene_txs: dict[str, dict[str, list[tuple[str, int, int, str]]]] = defaultdict(lambda: defaultdict(list))

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom, start, end, strand, attrs = parts[0], int(parts[3]), int(parts[4]), parts[6], parts[8]
            g = _GENE_ID_RE.search(attrs)
            t = _TRANSCRIPT_ID_RE.search(attrs)
            if g and t:
                gene_txs[g.group(1)][t.group(1)].append((chrom, start, end, strand))

    # For each gene pick the transcript with the most total exonic bp
    gene_best_exons: dict[str, list[tuple[str, int, int, str]]] = {}
    for gene_id, txs in gene_txs.items():
        gene_best_exons[gene_id] = max(
            txs.values(), key=lambda exons: sum(e - s + 1 for _, s, e, _ in exons)
        )

    fasta_path = _ensure_fasta(Path(fasta_gz), Path(fasta_path))
    print(f"Loading FASTA index from {fasta_path} …")
    fa = Fasta(str(fasta_path), sequence_always_upper=True, as_raw=True)
    fasta_chroms = set(fa.keys())

    def _resolve_chrom(chrom: str) -> str | None:
        if chrom in fasta_chroms:
            return chrom
        with_prefix = f"chr{chrom}"
        if with_prefix in fasta_chroms:
            return with_prefix
        return None

    result: dict[str, tuple[str, str]] = {}
    missing_chroms: set[str] = set()

    for gene_id, exons in gene_best_exons.items():
        chrom = exons[0][0]
        strand = exons[0][3]
        resolved = _resolve_chrom(chrom)
        if resolved is None:
            missing_chroms.add(chrom)
            continue

        sorted_exons = sorted(exons, key=lambda e: e[1])  # ascending genomic order
        if strand == "-":
            # RNA 5'→3' runs from highest to lowest genomic coordinate
            sorted_exons = sorted_exons[::-1]
            parts = [_revcomp(str(fa[resolved][s - 1:e])) for _, s, e, _ in sorted_exons]
        else:
            parts = [str(fa[resolved][s - 1:e]) for _, s, e, _ in sorted_exons]

        result[gene_id] = ("".join(parts), "")

    if missing_chroms:
        print(f"Warning: {len(missing_chroms)} chromosome(s) not found: {sorted(missing_chroms)[:10]}")

    return result


if __name__ == "__main__":
    import argparse
    import json

    _DEFAULTS = {
        "windowed":   "data/processed/body_sequences.json",
        "genomic":    "data/processed/body_sequences_genomic_full.json",
        "transcript": "data/processed/body_sequences_transcript.json",
    }

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtf", type=Path, default=_DEFAULT_GTF)
    parser.add_argument("--fasta", type=Path, default=_DEFAULT_FASTA)
    parser.add_argument("--fasta-gz", type=Path, default=_DEFAULT_FASTA_GZ)
    parser.add_argument("--sequence-type", choices=["windowed", "genomic", "transcript"],
                        default="windowed",
                        help="windowed=first/last 1000 bp (default), genomic=full span, transcript=spliced exons")
    parser.add_argument("--window", type=int, default=1000,
                        help="Window size in bp (only used with --sequence-type windowed)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (defaults per sequence type if omitted)")
    args = parser.parse_args()

    output = args.output or Path(_DEFAULTS[args.sequence_type])

    if args.sequence_type == "windowed":
        seqs = load_body_sequences(args.gtf, args.fasta, args.fasta_gz, args.window)
    elif args.sequence_type == "genomic":
        gene_bounds = parse_gtf(args.gtf)
        print(f"Parsed {len(gene_bounds)} genes from GTF.")
        seqs = extract_full_genomic_sequences(gene_bounds, args.fasta, args.fasta_gz)
    else:
        seqs = extract_spliced_sequences(args.gtf, args.fasta, args.fasta_gz)

    print(f"Extracted sequences for {len(seqs)} genes.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(seqs, f)
    print(f"Saved to {output}")
