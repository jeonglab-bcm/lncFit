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
from collections import Counter
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
        "xgboost_d3",
        "xgboost_d3_strength",
        "xgboost_d7",
        "xgboost_d7_strength",
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
        if name.endswith("_strength"):
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


def _metrics(y: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y[mask], predictions[mask])),
        "auprc": float(average_precision_score(y[mask], predictions[mask])),
    }


def _predict_values(model, X: np.ndarray, strength_model: bool) -> np.ndarray:
    if strength_model:
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
        "--sequences",
        type=Path,
        default=Path("data/processed/body_sequences_transcript.json"),
    )
    parser.add_argument("--embeddings", type=Path, default=None)
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
        args.depmap_dir,
        args.feature_set,
    )
    y = np.asarray([record.label for record in records], dtype=np.int8)
    strength = np.minimum(
        -np.log10(np.maximum([record.rra_pvalue for record in records], 1e-12)), 8.0
    )
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
            target = strength if _is_strength_model(name) else y
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
                        model, X[valid_cell], _is_strength_model(name)
                    )
            else:
                model = _make_model(name, args.seed + fold, positive_weight)
                model.fit(X[train_idx], target[train_idx])
                oof[valid_idx] = _predict_values(
                    model, X[valid_idx], _is_strength_model(name)
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
            "models": results,
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
                args.depmap_dir,
                args.feature_set,
            )
            if test_feature_names != feature_names:
                raise ValueError("Train/test feature columns do not match")
            test_predictions: dict[str, np.ndarray] = {}
            for name in predictions_by_model:
                model = _make_model(name, args.seed, positive_weight)
                target = strength if _is_strength_model(name) else y
                model.fit(X, target)
                test_predictions[name] = _predict_values(
                    model, X_test, _is_strength_model(name)
                )
            np.savez_compressed(
                args.output.with_suffix(".test.npz"),
                target=np.asarray([record.target for record in test_records]),
                cell_line=np.asarray([record.cell_line for record in test_records]),
                **test_predictions,
            )


if __name__ == "__main__":
    main()
