#!/usr/bin/env python3
"""Leakage-audited Day-14 lncRNA hit classifier.

This runner intentionally never loads Day-7 measurements.  It builds features
from pre-screen/static supplementary annotations, baseline expression,
transcript sequence descriptors, Celligner coordinates, and optional frozen
DNA language-model embeddings.  Model selection uses chromosome-grouped
out-of-fold predictions so all rows for an lncRNA remain on the same side of a
split.

The S1A column ``Number of cell lines showing essentiality`` is deliberately
excluded: it is derived from the same screen outcomes being predicted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.embeddings import load_embeddings
from lncfit.features import cell_embedding_block
from lncfit.screen_data import LncRnaRecord, load_jsonl


EVALUATED_CELL_LINES = {"HAP1", "K562", "MDA-MB-231", "THP1"}
ALL_CELL_LINES = ["HAP1", "HEK293FT", "K562", "MDA-MB-231", "THP1"]
BASES = "ACGT"
DEPMAP_MODEL_IDS = {
    "HAP1": "ACH-002475",
    "K562": "ACH-000551",
    "MDA-MB-231": "ACH-000768",
    "THP1": "ACH-000146",
}


def _safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.nan_to_num(values, nan=0.0), 0.0))


def _sequence_features(
    records: list[LncRnaRecord], sequence_path: Path
) -> tuple[np.ndarray, list[str]]:
    with sequence_path.open() as fh:
        raw: dict[str, list[str]] = json.load(fh)

    unique_targets = sorted({record.target for record in records})
    cache: dict[str, np.ndarray] = {}
    kmers = [a + b for a in BASES for b in BASES]
    names = [
        "seq_log_length",
        "seq_gc",
        "seq_entropy",
        "seq_ambiguous_fraction",
    ] + [f"seq_dinuc_{kmer}" for kmer in kmers]

    for target in unique_targets:
        seq = raw[target][0].upper()
        counts = Counter(base for base in seq if base in BASES)
        valid = sum(counts.values())
        probs = np.array([counts[base] / max(valid, 1) for base in BASES])
        entropy = float(-sum(p * math.log2(p) for p in probs if p > 0))
        dinuc = Counter(
            seq[i : i + 2]
            for i in range(max(len(seq) - 1, 0))
            if set(seq[i : i + 2]) <= set(BASES)
        )
        n_dinuc = max(sum(dinuc.values()), 1)
        cache[target] = np.array(
            [
                math.log1p(len(seq)),
                (counts["G"] + counts["C"]) / max(valid, 1),
                entropy,
                1.0 - valid / max(len(seq), 1),
                *[dinuc[kmer] / n_dinuc for kmer in kmers],
            ],
            dtype=np.float32,
        )

    return np.vstack([cache[record.target] for record in records]), names


def _supplementary_features(
    records: list[LncRnaRecord], mmc2_path: Path
) -> tuple[np.ndarray, list[str]]:
    s1a = pd.read_excel(mmc2_path, sheet_name="S1A", header=2).set_index("lncRNA")
    s1c = pd.read_excel(mmc2_path, sheet_name="S1C", header=2).set_index("lncRNA")
    s1e = pd.read_excel(mmc2_path, sheet_name="S1E", header=2).set_index("lncRNA")

    # These are static biological annotations.  Do not add the target-derived
    # "Number of cell lines showing essentiality" column here.
    numeric_cols = [
        "Transcript length",
        "Exons",
        "Tissue tau",
        "Time tau",
        "Dynamic",
        "Count dynamic tissues",
        "Distance to closest protein-coding gene",
        "Closest protein-coding gene Cas9 - DepMap score (23Q2, median)",
    ]
    categorical_cols = ["Strand", "Genomic class", "Age"]
    categories = {
        col: sorted(str(v) for v in s1a[col].dropna().unique())
        for col in categorical_cols
    }
    dynamic_tissues = sorted(
        {
            tissue.strip()
            for value in s1a["Dynamic tissues"].dropna().astype(str)
            for tissue in value.split(",")
            if tissue.strip()
        }
    )
    total_cols = list(s1c.columns)
    mrna_cols = list(s1e.columns)

    names = [
        "ann_log_transcript_length",
        "ann_log_exons",
        "ann_tissue_tau",
        "ann_time_tau",
        "ann_dynamic",
        "ann_count_dynamic_tissues",
        "ann_log_distance",
        "ann_neighbor_depmap_median",
        "ann_neighbor_depmap_missing",
        "ann_crispri_known",
        "ann_crispri_hit",
    ]
    for col in categorical_cols:
        names.extend(f"ann_{col}_{value}" for value in categories[col])
    names.extend(f"ann_dynamic_tissue_{value}" for value in dynamic_tissues)
    names.extend(f"expr_total_log_{col}" for col in total_cols)
    names.extend(f"expr_total_present_{col}" for col in total_cols)
    names.extend(f"expr_mrna_log_{col}" for col in mrna_cols)
    names.extend(f"expr_mrna_present_{col}" for col in mrna_cols)
    names.extend(
        [
            "expr_matched_total_base_log",
            "expr_matched_total_rfx_log",
            "expr_matched_mrna_rfx_log",
            "expr_matched_total_rfx_minus_base",
            "expr_matched_mrna_minus_total",
            "expr_matched_total_base_fraction_max",
            "expr_matched_total_rfx_fraction_max",
            "expr_matched_mrna_fraction_max",
        ]
    )

    matrix: list[list[float]] = []
    for record in records:
        ann = s1a.loc[record.target]
        numeric = pd.to_numeric(ann[numeric_cols], errors="coerce")
        depmap_median = float(numeric.iloc[7]) if pd.notna(numeric.iloc[7]) else 0.0
        row = [
            math.log1p(max(float(numeric.iloc[0]), 0.0)),
            math.log1p(max(float(numeric.iloc[1]), 0.0)),
            float(numeric.iloc[2]) if pd.notna(numeric.iloc[2]) else 0.0,
            float(numeric.iloc[3]) if pd.notna(numeric.iloc[3]) else 0.0,
            float(bool(ann["Dynamic"])),
            float(numeric.iloc[5]) if pd.notna(numeric.iloc[5]) else 0.0,
            math.log1p(abs(float(numeric.iloc[6])))
            if pd.notna(numeric.iloc[6])
            else 0.0,
            depmap_median,
            float(pd.isna(numeric.iloc[7])),
            float(pd.notna(ann["CRISPRi hit"])),
            float(bool(ann["CRISPRi hit"])) if pd.notna(ann["CRISPRi hit"]) else 0.0,
        ]
        for col in categorical_cols:
            value = str(ann[col]) if pd.notna(ann[col]) else ""
            row.extend(float(value == category) for category in categories[col])
        tissues = (
            {item.strip() for item in str(ann["Dynamic tissues"]).split(",")}
            if pd.notna(ann["Dynamic tissues"])
            else set()
        )
        row.extend(float(tissue in tissues) for tissue in dynamic_tissues)

        total = pd.to_numeric(s1c.loc[record.target, total_cols], errors="coerce").to_numpy(
            dtype=float
        )
        mrna = pd.to_numeric(s1e.loc[record.target, mrna_cols], errors="coerce").to_numpy(
            dtype=float
        )
        total_log = _safe_log1p(total)
        mrna_log = _safe_log1p(mrna)
        row.extend(total_log)
        row.extend((np.nan_to_num(total, nan=0.0) > 0).astype(float))
        row.extend(mrna_log)
        row.extend((np.nan_to_num(mrna, nan=0.0) > 0).astype(float))

        base_col = record.cell_line
        rfx_col = f"{record.cell_line} RfxCas13d"
        base = float(s1c.at[record.target, base_col]) if base_col in s1c.columns else 0.0
        total_rfx = (
            float(s1c.at[record.target, rfx_col]) if rfx_col in s1c.columns else 0.0
        )
        mrna_rfx = float(s1e.at[record.target, rfx_col]) if rfx_col in s1e.columns else 0.0
        base_log, total_rfx_log, mrna_rfx_log = np.log1p(
            np.maximum([base, total_rfx, mrna_rfx], 0.0)
        )
        row.extend(
            [
                base_log,
                total_rfx_log,
                mrna_rfx_log,
                total_rfx_log - base_log,
                mrna_rfx_log - total_rfx_log,
                base_log / max(float(total_log.max()), 1e-6),
                total_rfx_log / max(float(total_log.max()), 1e-6),
                mrna_rfx_log / max(float(mrna_log.max()), 1e-6),
            ]
        )
        matrix.append(row)

    return np.asarray(matrix, dtype=np.float32), names


def _normalized_entropy(values: np.ndarray) -> float:
    values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probabilities = values[values > 0] / total
    return float(
        -np.sum(probabilities * np.log(probabilities))
        / math.log(max(len(values), 2))
    )


def _cross_cell_summaries(
    values: np.ndarray, matched_index: int
) -> list[float]:
    values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
    logged = np.log1p(values)
    matched = float(values[matched_index])
    matched_log = float(logged[matched_index])
    others = np.delete(values, matched_index)
    other_logs = np.delete(logged, matched_index)
    mean_other = float(others.mean())
    maximum = float(values.max())
    mean = float(values.mean())
    std = float(values.std())
    tau = (
        float(np.sum(1.0 - values / maximum) / max(len(values) - 1, 1))
        if maximum > 0
        else 0.0
    )
    return [
        matched_log,
        float(other_logs.mean()),
        float(other_logs.max()),
        math.log2((matched + 0.1) / (mean_other + 0.1)),
        matched / max(float(values.sum()), 1e-6),
        float((values <= matched).mean()),
        std / max(mean, 1e-6),
        tau,
        _normalized_entropy(values),
        float((values > 0).sum()),
        float((values >= 1).sum()),
    ]


def _expression_specificity_features(
    records: list[LncRnaRecord], mmc2_path: Path
) -> tuple[np.ndarray, list[str]]:
    """Explicit cross-cell expression contrasts from pre-screen RNA-seq."""
    total = pd.read_excel(mmc2_path, sheet_name="S1C", header=2).set_index("lncRNA")
    mrna = pd.read_excel(mmc2_path, sheet_name="S1E", header=2).set_index("lncRNA")
    total_base_cols = ALL_CELL_LINES
    total_rfx_cols = [
        cell if cell == "MDA-MB-231" else f"{cell} RfxCas13d"
        for cell in ALL_CELL_LINES
    ]
    mrna_rfx_cols = [f"{cell} RfxCas13d" for cell in ALL_CELL_LINES]
    suffixes = [
        "matched_log",
        "other_mean_log",
        "other_max_log",
        "matched_vs_other_log2_ratio",
        "matched_fraction",
        "matched_within_gene_percentile",
        "cross_cell_cv",
        "cross_cell_tau",
        "cross_cell_entropy",
        "expressing_cell_count",
        "cell_count_tpm_ge_1",
    ]
    names = [
        f"expr_specificity_{modality}_{suffix}"
        for modality in ("total_base", "total_rfx", "mrna_rfx")
        for suffix in suffixes
    ]
    names.extend(
        [
            "expr_specificity_total_base_global_percentile",
            "expr_specificity_total_rfx_global_percentile",
            "expr_specificity_mrna_rfx_global_percentile",
        ]
    )

    total_base_percentiles = total[total_base_cols].rank(pct=True)
    total_rfx_percentiles = total[total_rfx_cols].rank(pct=True)
    mrna_rfx_percentiles = mrna[mrna_rfx_cols].rank(pct=True)
    matrix: list[list[float]] = []
    for record in records:
        matched_index = ALL_CELL_LINES.index(record.cell_line)
        base_values = pd.to_numeric(
            total.loc[record.target, total_base_cols], errors="coerce"
        ).to_numpy(dtype=float)
        total_rfx_values = pd.to_numeric(
            total.loc[record.target, total_rfx_cols], errors="coerce"
        ).to_numpy(dtype=float)
        mrna_rfx_values = pd.to_numeric(
            mrna.loc[record.target, mrna_rfx_cols], errors="coerce"
        ).to_numpy(dtype=float)
        row = [
            *_cross_cell_summaries(base_values, matched_index),
            *_cross_cell_summaries(total_rfx_values, matched_index),
            *_cross_cell_summaries(mrna_rfx_values, matched_index),
            float(
                total_base_percentiles.at[
                    record.target, total_base_cols[matched_index]
                ]
            ),
            float(
                total_rfx_percentiles.at[
                    record.target, total_rfx_cols[matched_index]
                ]
            ),
            float(
                mrna_rfx_percentiles.at[
                    record.target, mrna_rfx_cols[matched_index]
                ]
            ),
        ]
        matrix.append(row)
    return np.asarray(matrix, dtype=np.float32), names


_GTF_GENE_RE = re.compile(r"gene_id\s+(\S+?);")
_GTF_TRANSCRIPT_RE = re.compile(r"transcript_id\s+(\S+?);")


def _transcript_architecture_features(
    records: list[LncRnaRecord], gtf_path: Path
) -> tuple[np.ndarray, list[str]]:
    """Summarize isoform/exon architecture without using screen measurements."""
    wanted = {record.target for record in records}
    exons: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with gtf_path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            gene_match = _GTF_GENE_RE.search(fields[8])
            transcript_match = _GTF_TRANSCRIPT_RE.search(fields[8])
            if not gene_match or not transcript_match:
                continue
            gene = gene_match.group(1)
            if gene not in wanted:
                continue
            exons[gene][transcript_match.group(1)].append(
                (int(fields[3]), int(fields[4]))
            )

    names = [
        "architecture_log_transcript_count",
        "architecture_log_longest_exonic_length",
        "architecture_log_mean_exonic_length",
        "architecture_log_shortest_exonic_length",
        "architecture_transcript_length_cv",
        "architecture_longest_to_mean_length",
        "architecture_mean_exon_count",
        "architecture_max_exon_count",
        "architecture_exon_count_cv",
        "architecture_log_union_exonic_bp",
        "architecture_log_genomic_span",
        "architecture_union_to_longest_ratio",
    ]
    cache: dict[str, np.ndarray] = {}
    for gene in wanted:
        transcripts = exons.get(gene, {})
        lengths = np.asarray(
            [
                sum(end - start + 1 for start, end in transcript_exons)
                for transcript_exons in transcripts.values()
            ],
            dtype=float,
        )
        exon_counts = np.asarray(
            [len(transcript_exons) for transcript_exons in transcripts.values()],
            dtype=float,
        )
        intervals = sorted(
            interval
            for transcript_exons in transcripts.values()
            for interval in transcript_exons
        )
        merged: list[list[int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        union_bp = sum(end - start + 1 for start, end in merged)
        genomic_span = (
            max(end for _, end in intervals) - min(start for start, _ in intervals) + 1
            if intervals
            else 0
        )
        if lengths.size:
            mean_length = float(lengths.mean())
            longest = float(lengths.max())
            shortest = float(lengths.min())
            length_cv = float(lengths.std() / max(mean_length, 1e-6))
            mean_exons = float(exon_counts.mean())
            max_exons = float(exon_counts.max())
            exon_cv = float(exon_counts.std() / max(mean_exons, 1e-6))
        else:
            mean_length = longest = shortest = length_cv = 0.0
            mean_exons = max_exons = exon_cv = 0.0
        cache[gene] = np.asarray(
            [
                math.log1p(len(transcripts)),
                math.log1p(longest),
                math.log1p(mean_length),
                math.log1p(shortest),
                length_cv,
                longest / max(mean_length, 1e-6),
                mean_exons,
                max_exons,
                exon_cv,
                math.log1p(union_bp),
                math.log1p(genomic_span),
                union_bp / max(longest, 1e-6),
            ],
            dtype=np.float32,
        )
    return np.vstack([cache[record.target] for record in records]), names


def _max_homopolymer(sequence: str) -> int:
    if not sequence:
        return 0
    maximum = run = 1
    for left, right in zip(sequence, sequence[1:], strict=False):
        run = run + 1 if left == right else 1
        maximum = max(maximum, run)
    return maximum


def _self_complementarity(sequence: str) -> int:
    reverse_complement = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    maximum = 0
    length = len(sequence)
    for shift in range(-length + 1, length):
        run = 0
        for index in range(length):
            other = index + shift
            if 0 <= other < length and sequence[index] == reverse_complement[other]:
                run += 1
                maximum = max(maximum, run)
            else:
                run = 0
    return maximum


def _guide_design_features(
    records: list[LncRnaRecord], mmc2_path: Path, sequence_path: Path
) -> tuple[np.ndarray, list[str]]:
    """Aggregate pre-screen guide design and longest-isoform coverage."""
    guides = pd.read_excel(
        mmc2_path, sheet_name="S1B", header=2, dtype=str
    ).dropna(subset=["Target", "Sequence (5' - 3')"])
    wanted = {record.target for record in records}
    guides = guides[guides["Target"].isin(wanted)]
    guides_by_target = {
        target: group["Sequence (5' - 3')"].str.upper().tolist()
        for target, group in guides.groupby("Target")
    }
    with sequence_path.open() as fh:
        raw_sequences: dict[str, list[str]] = json.load(fh)

    metrics = [
        "gc",
        "entropy",
        "homopolymer",
        "complexity3",
        "self_complementarity",
        "exact_match_count",
        "position",
        "local_gc",
        "local_entropy",
    ]
    names = ["guide_count"]
    names.extend(
        f"guide_{metric}_{stat}"
        for metric in metrics
        for stat in ("mean", "std", "min", "max")
    )
    names.extend(
        [
            "guide_exact_match_fraction",
            "guide_unique_match_fraction",
            "guide_high_quality_fraction",
            "guide_position_range",
            "guide_position_unique_fraction",
        ]
    )
    names.extend(f"guide_base_{position}_{base}" for position in range(23) for base in BASES)
    names.extend(f"guide_dinucleotide_{left}{right}" for left in BASES for right in BASES)

    complement = str.maketrans("ACGT", "TGCA")
    cache: dict[str, np.ndarray] = {}
    for target in wanted:
        target_guides = guides_by_target.get(target, [])
        transcript = raw_sequences[target][0].upper()
        guide_rows: list[list[float]] = []
        base_counts = np.zeros((23, 4), dtype=float)
        dinucleotide_counts = Counter()
        positions: list[float] = []
        high_quality = 0
        exact_matches = 0
        unique_matches = 0
        for guide in target_guides:
            guide = guide.upper()
            reverse_complement = guide.translate(complement)[::-1]
            counts = Counter(base for base in guide if base in BASES)
            valid = sum(counts.values())
            probabilities = [
                counts[base] / max(valid, 1) for base in BASES if counts[base]
            ]
            entropy = float(-sum(p * math.log2(p) for p in probabilities))
            kmers3 = {
                guide[index : index + 3]
                for index in range(max(len(guide) - 2, 0))
            }
            complexity = len(kmers3) / max(len(guide) - 2, 1)
            matches = transcript.count(reverse_complement)
            position = transcript.find(reverse_complement)
            normalized_position = (
                position / max(len(transcript) - len(guide), 1)
                if position >= 0
                else 0.0
            )
            if position >= 0:
                context = transcript[max(0, position - 50) : position + len(guide) + 50]
                context_counts = Counter(base for base in context if base in BASES)
                context_valid = sum(context_counts.values())
                local_gc = (
                    context_counts["G"] + context_counts["C"]
                ) / max(context_valid, 1)
                context_probabilities = [
                    context_counts[base] / max(context_valid, 1)
                    for base in BASES
                    if context_counts[base]
                ]
                local_entropy = float(
                    -sum(p * math.log2(p) for p in context_probabilities)
                )
                positions.append(normalized_position)
            else:
                local_gc = local_entropy = 0.0
            gc = (counts["G"] + counts["C"]) / max(valid, 1)
            homopolymer = _max_homopolymer(guide)
            self_complementarity = _self_complementarity(guide)
            exact_matches += int(matches > 0)
            unique_matches += int(matches == 1)
            high_quality += int(
                0.35 <= gc <= 0.65
                and homopolymer < 4
                and self_complementarity < 7
                and matches == 1
            )
            guide_rows.append(
                [
                    gc,
                    entropy,
                    homopolymer,
                    complexity,
                    self_complementarity,
                    matches,
                    normalized_position,
                    local_gc,
                    local_entropy,
                ]
            )
            for index, base in enumerate(guide[:23]):
                if base in BASES:
                    base_counts[index, BASES.index(base)] += 1
            dinucleotide_counts.update(
                guide[index : index + 2]
                for index in range(max(len(guide) - 1, 0))
                if set(guide[index : index + 2]) <= set(BASES)
            )

        n_guides = len(target_guides)
        values = np.asarray(guide_rows, dtype=float)
        row: list[float] = [float(n_guides)]
        for metric_index in range(len(metrics)):
            column = values[:, metric_index] if values.size else np.zeros(1)
            row.extend(
                [
                    float(column.mean()),
                    float(column.std()),
                    float(column.min()),
                    float(column.max()),
                ]
            )
        row.extend(
            [
                exact_matches / max(n_guides, 1),
                unique_matches / max(n_guides, 1),
                high_quality / max(n_guides, 1),
                max(positions) - min(positions) if positions else 0.0,
                len(set(positions)) / max(n_guides, 1),
            ]
        )
        row.extend((base_counts / max(n_guides, 1)).ravel())
        total_dinucleotides = max(sum(dinucleotide_counts.values()), 1)
        row.extend(
            dinucleotide_counts[left + right] / total_dinucleotides
            for left in BASES
            for right in BASES
        )
        cache[target] = np.asarray(row, dtype=np.float32)
    return np.vstack([cache[record.target] for record in records]), names


def _guide_flank_features(
    records: list[LncRnaRecord], mmc2_path: Path, sequence_path: Path
) -> tuple[np.ndarray, list[str]]:
    """Aggregate immediate target-site context for the pre-designed guides."""
    guides = pd.read_excel(
        mmc2_path, sheet_name="S1B", header=2, dtype=str
    ).dropna(subset=["Target", "Sequence (5' - 3')"])
    wanted = {record.target for record in records}
    guides = guides[guides["Target"].isin(wanted)]
    guides_by_target = {
        target: group["Sequence (5' - 3')"].str.upper().tolist()
        for target, group in guides.groupby("Target")
    }
    with sequence_path.open() as fh:
        raw_sequences: dict[str, list[str]] = json.load(fh)

    offsets = [*range(-5, 0), *range(1, 6)]
    names = [
        f"guide_flank_{offset:+d}_{base}"
        for offset in offsets
        for base in BASES
    ]
    complement = str.maketrans("ACGT", "TGCA")
    cache: dict[str, np.ndarray] = {}
    for target in wanted:
        transcript = raw_sequences[target][0].upper()
        target_guides = guides_by_target.get(target, [])
        counts = np.zeros((len(offsets), len(BASES)), dtype=float)
        for guide in target_guides:
            reverse_complement = guide.translate(complement)[::-1]
            position = transcript.find(reverse_complement)
            if position < 0:
                continue
            for offset_index, offset in enumerate(offsets):
                transcript_index = (
                    position + offset
                    if offset < 0
                    else position + len(guide) + offset - 1
                )
                if 0 <= transcript_index < len(transcript):
                    base = transcript[transcript_index]
                    if base in BASES:
                        counts[offset_index, BASES.index(base)] += 1
        cache[target] = (
            counts / max(len(target_guides), 1)
        ).ravel().astype(np.float32)
    return np.vstack([cache[record.target] for record in records]), names


def _load_depmap_rows(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Load only the four relevant DepMap rows without materializing the huge CSV."""
    wanted = set(DEPMAP_MODEL_IDS.values())
    rows: dict[str, np.ndarray] = {}
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        gene_index: dict[str, int] = {}
        for index, label in enumerate(header[1:]):
            symbol = re.sub(r"\s+\(\d+\)$", "", label)
            gene_index.setdefault(symbol, index)
        for row in reader:
            model_id = row[0]
            if model_id in wanted:
                rows[model_id] = np.asarray(
                    [float(value) if value else np.nan for value in row[1:]],
                    dtype=np.float32,
                )
                if len(rows) == len(wanted):
                    break
    return rows, gene_index


def _depmap_features(
    records: list[LncRnaRecord], depmap_dir: Path
) -> tuple[np.ndarray, list[str]]:
    effect_rows, effect_index = _load_depmap_rows(depmap_dir / "CRISPRGeneEffect.csv")
    expression_rows, expression_index = _load_depmap_rows(
        depmap_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    )
    ordered_cells = list(DEPMAP_MODEL_IDS)
    names = [
        "depmap_matched_neighbor_effect",
        "depmap_matched_neighbor_expression",
        "depmap_matched_neighbor_effect_missing",
        "depmap_matched_neighbor_expression_missing",
    ]
    names.extend(f"depmap_neighbor_effect_{cell}" for cell in ordered_cells)
    names.extend(f"depmap_neighbor_expression_{cell}" for cell in ordered_cells)
    names.extend(
        f"depmap_neighbor_effect_{stat}" for stat in ("mean", "min", "max", "std")
    )
    names.extend(
        f"depmap_neighbor_expression_{stat}" for stat in ("mean", "min", "max", "std")
    )

    matrix = np.zeros((len(records), len(names)), dtype=np.float32)
    for row_index, record in enumerate(records):
        effect_col = effect_index.get(record.closest_pc_gene)
        expression_col = expression_index.get(record.closest_pc_gene)
        effects = np.asarray(
            [
                effect_rows.get(DEPMAP_MODEL_IDS[cell], np.empty(0))[effect_col]
                if effect_col is not None
                and DEPMAP_MODEL_IDS[cell] in effect_rows
                else np.nan
                for cell in ordered_cells
            ],
            dtype=np.float32,
        )
        expressions = np.asarray(
            [
                expression_rows.get(DEPMAP_MODEL_IDS[cell], np.empty(0))[expression_col]
                if expression_col is not None
                and DEPMAP_MODEL_IDS[cell] in expression_rows
                else np.nan
                for cell in ordered_cells
            ],
            dtype=np.float32,
        )
        matched = (
            ordered_cells.index(record.cell_line)
            if record.cell_line in ordered_cells
            else None
        )
        matched_effect = effects[matched] if matched is not None else np.nan
        matched_expression = expressions[matched] if matched is not None else np.nan
        valid_effects = effects[np.isfinite(effects)]
        valid_expressions = expressions[np.isfinite(expressions)]
        effect_stats = (
            [
                valid_effects.mean(),
                valid_effects.min(),
                valid_effects.max(),
                valid_effects.std(),
            ]
            if valid_effects.size
            else [0.0] * 4
        )
        expression_stats = (
            [
                valid_expressions.mean(),
                valid_expressions.min(),
                valid_expressions.max(),
                valid_expressions.std(),
            ]
            if valid_expressions.size
            else [0.0] * 4
        )
        matrix[row_index] = np.asarray(
            [
                np.nan_to_num(matched_effect),
                np.nan_to_num(matched_expression),
                float(not np.isfinite(matched_effect)),
                float(not np.isfinite(matched_expression)),
                *np.nan_to_num(effects),
                *np.nan_to_num(expressions),
                *effect_stats,
                *expression_stats,
            ],
            dtype=np.float32,
        )
    return matrix, names


def build_features(
    records: list[LncRnaRecord],
    mmc2_path: Path,
    sequence_path: Path,
    embeddings_path: Path | None,
    gtf_path: Path = Path("data/raw/human.lncRNA.hg19.gtf"),
    feature_blocks: tuple[str, ...] = (),
    depmap_dir: Path | None = None,
    feature_set: str = "multimodal",
) -> tuple[np.ndarray, list[str]]:
    if feature_set == "dnabert_base" and embeddings_path is None:
        raise ValueError("feature_set='dnabert_base' requires --embeddings")
    blocks: list[np.ndarray] = []
    names: list[str] = []

    if feature_set != "dnabert_base":
        supplementary, supplementary_names = _supplementary_features(records, mmc2_path)
        blocks.append(supplementary)
        names.extend(supplementary_names)

        sequence, sequence_names = _sequence_features(records, sequence_path)
        blocks.append(sequence)
        names.extend(sequence_names)

        if "expression_specificity" in feature_blocks:
            expression, expression_names = _expression_specificity_features(
                records, mmc2_path
            )
            blocks.append(expression)
            names.extend(expression_names)

        if "transcript_architecture" in feature_blocks:
            architecture, architecture_names = _transcript_architecture_features(
                records, gtf_path
            )
            blocks.append(architecture)
            names.extend(architecture_names)

        guide_blocks = {
            block
            for block in feature_blocks
            if block
            in {
                "guide_qc",
                "guide_sequence",
                "guide_context",
                "guide_design",
            }
        }
        if guide_blocks:
            guides, guide_names = _guide_design_features(
                records, mmc2_path, sequence_path
            )
            if "guide_design" not in guide_blocks:
                selected = []
                for index, name in enumerate(guide_names):
                    is_qc = (
                        name == "guide_count"
                        or "exact_match" in name
                        or "unique_match" in name
                        or "high_quality" in name
                        or "position_" in name
                    )
                    is_sequence = (
                        any(
                            token in name
                            for token in (
                                "_gc_",
                                "_entropy_",
                                "_homopolymer_",
                                "_complexity3_",
                                "_self_complementarity_",
                                "guide_base_",
                                "guide_dinucleotide_",
                            )
                        )
                        and "local_" not in name
                    )
                    is_context = "local_gc" in name or "local_entropy" in name
                    if (
                        ("guide_qc" in guide_blocks and is_qc)
                        or ("guide_sequence" in guide_blocks and is_sequence)
                        or ("guide_context" in guide_blocks and is_context)
                    ):
                        selected.append(index)
                guides = guides[:, selected]
                guide_names = [guide_names[index] for index in selected]
            blocks.append(guides)
            names.extend(guide_names)

        if "guide_flanks" in feature_blocks:
            guide_flanks, guide_flank_names = _guide_flank_features(
                records, mmc2_path, sequence_path
            )
            blocks.append(guide_flanks)
            names.extend(guide_flank_names)

    celligner, celligner_names = cell_embedding_block(records, dim=70)
    blocks.append(celligner)
    names.extend(celligner_names)

    cell_lines = sorted({record.cell_line for record in records})
    blocks.append(
        np.asarray(
            [[float(record.cell_line == cell) for cell in cell_lines] for record in records],
            dtype=np.float32,
        )
    )
    names.extend(f"cell_{cell}" for cell in cell_lines)

    if feature_set == "dnabert_base":
        blocks.append(
            np.asarray(
                [
                    [
                        float(record.distance_to_closest_pc_gene)
                        if record.distance_to_closest_pc_gene is not None
                        else 0.0
                    ]
                    for record in records
                ],
                dtype=np.float32,
            )
        )
        names.append("distance_to_closest_pc_gene")

    if depmap_dir is not None:
        depmap, depmap_names = _depmap_features(records, depmap_dir)
        blocks.append(depmap)
        names.extend(depmap_names)

    if embeddings_path is not None:
        embedding_matrix, embedding_index = load_embeddings(embeddings_path)
        missing = sorted({record.target for record in records} - embedding_index.keys())
        if missing:
            raise ValueError(
                f"DNABERT embeddings miss {len(missing):,} target(s), e.g. {missing[:3]}"
            )
        blocks.append(
            np.vstack([embedding_matrix[embedding_index[r.target]] for r in records]).astype(
                np.float32
            )
        )
        names.extend(f"dnabert_{i}" for i in range(embedding_matrix.shape[1]))

    return np.hstack(blocks), names


def _make_model(name: str, seed: int, positive_weight: float):
    if name.startswith("per_cell_"):
        name = name.removeprefix("per_cell_")
    if name in {
        "xgboost",
        "xgboost_strength",
        "xgboost_effect",
        "xgboost_d3",
        "xgboost_d3_strength",
        "xgboost_d3_effect",
        "xgboost_d7",
        "xgboost_d7_strength",
        "xgboost_d7_effect",
    }:
        from xgboost import XGBClassifier, XGBRegressor

        max_depth = 3 if "_d3" in name else 7 if "_d7" in name else 5
        common = dict(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=max_depth,
            min_child_weight=3,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=8,
            random_state=seed,
        )
        if name.endswith(("_strength", "_effect")):
            return XGBRegressor(
                **common,
                objective="reg:pseudohubererror",
                eval_metric="mae",
            )
        return XGBClassifier(
            **common,
            objective="binary:logistic",
            eval_metric="aucpr",
        )
    if name == "xgboost_weighted":
        model = _make_model("xgboost", seed, positive_weight)
        model.set_params(scale_pos_weight=positive_weight)
        return model
    if name in {"lightgbm", "lightgbm_strength"}:
        from lightgbm import LGBMClassifier, LGBMRegressor

        common = dict(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            verbosity=-1,
            n_jobs=8,
            random_state=seed,
        )
        if name == "lightgbm_strength":
            return LGBMRegressor(**common, objective="huber")
        return LGBMClassifier(**common)
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=600,
            min_samples_leaf=3,
            max_features=0.7,
            class_weight="balanced",
            n_jobs=8,
            random_state=seed,
        )
    if name == "svm_nystroem":
        from lncfit.classifiers.svm import SVMClassifier

        return SVMClassifier(
            kernel="rbf",
            C=0.1,
            kernel_approx=2000,
            class_weight="balanced",
            seed=seed,
        )
    raise ValueError(f"Unknown model: {name}")


def _make_gene_model(seed: int):
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=8,
        random_state=seed,
    )


def _fit_gene_propensity_fold(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    strength: np.ndarray,
    records: list[LncRnaRecord],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Fit one training-chromosome-only gene propensity model."""
    train_by_target: dict[str, list[int]] = {}
    valid_by_target: dict[str, list[int]] = {}
    for index in train_idx:
        if records[index].cell_line in EVALUATED_CELL_LINES:
            train_by_target.setdefault(records[index].target, []).append(int(index))
    for index in valid_idx:
        if records[index].cell_line in EVALUATED_CELL_LINES:
            valid_by_target.setdefault(records[index].target, []).append(int(index))

    train_targets = sorted(train_by_target)
    valid_targets = sorted(valid_by_target)
    X_train = np.vstack(
        [X[train_by_target[target]].mean(axis=0) for target in train_targets]
    )
    X_valid = np.vstack(
        [X[valid_by_target[target]].mean(axis=0) for target in valid_targets]
    )
    if name == "gene_count_xgboost":
        target = np.asarray(
            [y[train_by_target[value]].sum() for value in train_targets], dtype=float
        )
    elif name == "gene_max_strength_xgboost":
        target = np.asarray(
            [strength[train_by_target[value]].max() for value in train_targets],
            dtype=float,
        )
    else:
        raise ValueError(f"Unknown gene model: {name}")

    model = _make_gene_model(seed)
    model.fit(X_train, target)
    gene_predictions = dict(zip(valid_targets, model.predict(X_valid), strict=True))
    return np.asarray(
        [gene_predictions.get(records[index].target, 0.0) for index in valid_idx],
        dtype=float,
    )


def _is_strength_model(name: str) -> bool:
    return name.removeprefix("per_cell_").endswith("_strength")


def _is_effect_model(name: str) -> bool:
    return name.removeprefix("per_cell_").endswith("_effect")


def _is_regression_model(name: str) -> bool:
    return _is_strength_model(name) or _is_effect_model(name)


def _metrics(y: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y[mask], predictions[mask])),
        "auprc": float(average_precision_score(y[mask], predictions[mask])),
    }


def _predict_values(model, X: np.ndarray, regression_model: bool) -> np.ndarray:
    if regression_model:
        return np.asarray(model.predict(X), dtype=float)
    probabilities = np.asarray(model.predict_proba(X))
    return (
        probabilities[:, 1].astype(float)
        if probabilities.ndim == 2
        else probabilities.astype(float)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", type=Path, default=Path("data/processed/train_lncrna_day14_chrom1.jsonl.gz")
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=None,
        help="Optional held-out records. When supplied, fit each selected model on "
             "all training rows and save unscored predictions beside --output.",
    )
    parser.add_argument("--mmc2", type=Path, default=Path("data/raw/mmc2.xlsx"))
    parser.add_argument(
        "--gtf", type=Path, default=Path("data/raw/human.lncRNA.hg19.gtf")
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        default=Path("data/processed/body_sequences_transcript.json"),
    )
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument(
        "--feature-blocks",
        nargs="*",
        choices=[
            "expression_specificity",
            "transcript_architecture",
            "guide_qc",
            "guide_sequence",
            "guide_context",
            "guide_design",
            "guide_flanks",
        ],
        default=[],
        help="Optional pre-screen feature blocks added to the multimodal core.",
    )
    parser.add_argument("--depmap-dir", type=Path, default=None)
    parser.add_argument(
        "--feature-set",
        choices=["multimodal", "dnabert_base"],
        default="multimodal",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "xgboost_weighted", "lightgbm", "histgb"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.test is not None and args.output is None:
        parser.error("--test requires --output")

    records: list[LncRnaRecord] = load_jsonl(args.train, record_cls=LncRnaRecord)
    if any(record.day != 14 for record in records):
        raise ValueError("Input contains a non-Day-14 target record")

    X, feature_names = build_features(
        records,
        args.mmc2,
        args.sequences,
        args.embeddings,
        args.gtf,
        tuple(args.feature_blocks),
        args.depmap_dir,
        args.feature_set,
    )
    y = np.asarray([record.label for record in records], dtype=np.int8)
    strength = np.minimum(
        -np.log10(np.maximum([record.rra_pvalue for record in records], 1e-12)), 8.0
    )
    effect = np.maximum(-np.asarray([record.fold_change for record in records]), 0.0)
    groups = np.asarray([record.chrom for record in records])
    evaluated = np.asarray([record.cell_line in EVALUATED_CELL_LINES for record in records])
    positive_weight = float((y == 0).sum() / (y == 1).sum())
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    results: list[dict[str, float | str]] = []
    predictions_by_model: dict[str, np.ndarray] = {}
    for name in args.models:
        oof = np.full(len(records), np.nan, dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(
            splitter.split(X, y, groups), start=1
        ):
            if name.startswith("gene_"):
                oof[valid_idx] = _fit_gene_propensity_fold(
                    name,
                    X,
                    y,
                    strength,
                    records,
                    train_idx,
                    valid_idx,
                    args.seed + fold,
                )
                continue
            if _is_strength_model(name):
                target = strength
            elif _is_effect_model(name):
                target = effect
            else:
                target = y
            if name.startswith("per_cell_"):
                for cell_line in sorted({record.cell_line for record in records}):
                    train_cell = train_idx[
                        np.asarray(
                            [records[index].cell_line == cell_line for index in train_idx]
                        )
                    ]
                    valid_cell = valid_idx[
                        np.asarray(
                            [records[index].cell_line == cell_line for index in valid_idx]
                        )
                    ]
                    model = _make_model(name, args.seed + fold, positive_weight)
                    model.fit(X[train_cell], target[train_cell])
                    oof[valid_cell] = _predict_values(
                        model, X[valid_cell], _is_regression_model(name)
                    )
            else:
                model = _make_model(name, args.seed + fold, positive_weight)
                model.fit(X[train_idx], target[train_idx])
                oof[valid_idx] = _predict_values(
                    model, X[valid_idx], _is_regression_model(name)
                )
        assert np.isfinite(oof).all()
        predictions_by_model[name] = oof
        row = {"model": name, **_metrics(y, oof, evaluated)}
        results.append(row)
        print(json.dumps(row), flush=True)

    # Rank averaging is scale-invariant and is selected strictly from OOF predictions.
    if len(predictions_by_model) > 1:
        ranks = np.vstack(
            [
                rankdata(predictions) / len(predictions)
                for predictions in predictions_by_model.values()
            ]
        )
        ensemble = ranks.mean(axis=0)
        row = {"model": "rank_mean_all", **_metrics(y, ensemble, evaluated)}
        results.append(row)
        print(json.dumps(row), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input": str(args.train),
            "feature_count": len(feature_names),
            "features": feature_names,
            "feature_blocks": args.feature_blocks,
            "models": results,
            "day0_features_used": False,
            "day7_features_used": False,
        }
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        np.savez_compressed(
            args.output.with_suffix(".oof.npz"),
            target=np.asarray([record.target for record in records]),
            cell_line=np.asarray([record.cell_line for record in records]),
            y=y,
            **predictions_by_model,
        )

        if args.test is not None:
            if any(
                name.startswith(("gene_", "per_cell_"))
                for name in predictions_by_model
            ):
                raise ValueError(
                    "--test currently supports pooled row models only; "
                    "gene/per-cell models were experimental CV ablations"
                )
            test_records: list[LncRnaRecord] = load_jsonl(
                args.test, record_cls=LncRnaRecord
            )
            if any(record.day != 14 for record in test_records):
                raise ValueError("Test input contains a non-Day-14 target record")
            X_test, test_feature_names = build_features(
                test_records,
                args.mmc2,
                args.sequences,
                args.embeddings,
                args.gtf,
                tuple(args.feature_blocks),
                args.depmap_dir,
                args.feature_set,
            )
            if test_feature_names != feature_names:
                raise ValueError("Train/test feature columns do not match")
            test_predictions: dict[str, np.ndarray] = {}
            for name in predictions_by_model:
                model = _make_model(name, args.seed, positive_weight)
                if _is_strength_model(name):
                    target = strength
                elif _is_effect_model(name):
                    target = effect
                else:
                    target = y
                model.fit(X, target)
                test_predictions[name] = _predict_values(
                    model, X_test, _is_regression_model(name)
                )
            np.savez_compressed(
                args.output.with_suffix(".test.npz"),
                target=np.asarray([record.target for record in test_records]),
                cell_line=np.asarray([record.cell_line for record in test_records]),
                **test_predictions,
            )


if __name__ == "__main__":
    main()
