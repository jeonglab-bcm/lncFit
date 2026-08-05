"""Score one lncRNA (sequence + cell line) with a trained RRA-hit classifier.

Inspection tool for the model produced by scripts/train_lncrna_xgboost.py: takes a
single lncRNA plus a cell line, builds the exact same feature vector the training
code builds, and prints P(essential) in [0, 1]. Nothing else in the repo imports
this module -- it exists to make one prediction legible end to end.

Usage (from project root; gene IDs are the GTF's Hum_XLOC_* form, not Ensembl):
  uv run python scripts/predict_lncrna.py --gene-id Hum_XLOC_000004 --cell-line THP1
  uv run python scripts/predict_lncrna.py --sequence ACGTACGT... --cell-line K562
  uv run python scripts/predict_lncrna.py --gene-id Hum_XLOC_000004 --all-cell-lines --show-vector

The feature vector is k-mer frequencies (column order fixed by the saved vocab, NOT
refit here) + a 5-way cell-line one-hot [+ distance_to_closest_pc_gene, if the model
was trained with it]. k is inferred from the vocab, so --k is never needed.

The score is inflated by the scale_pos_weight class balancing used in training, so
it is a ranking score rather than a calibrated probability: rank lncRNAs by it and
pick a threshold on validation data instead of assuming 0.5.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent.parent))

# Imported rather than re-listed so the one-hot order can never drift from the
# order build_lncrna_features() used at training time.
from lncfit.features import _CELL_LINES as CELL_LINES
from lncfit.features import build_lncrna_features
from lncfit.screen_data import LncRnaRecord

_DEFAULT_MODEL = "data/model/xgboost_lncrna_day14_k6.ubj"
_DEFAULT_SEQUENCES = "data/processed/body_sequences_transcript.json"


def _sidecar(model_path: Path, suffix: str) -> Path:
    """Map .../xgboost_lncrna_day14_k6.ubj -> .../xgboost_lncrna_day14_k6_<suffix>.json.

    train_lncrna_xgboost.py writes the vocab and params next to the model under this
    naming convention, so the caller only ever has to pass --model.
    """
    return model_path.with_name(f"{model_path.stem}_{suffix}.json")


def _load_transcript_sequences(path: str) -> dict[str, str]:
    """Load {gene_id: [seq, ""]} from lncfit.sequence and flatten to {gene_id: seq}."""
    with open(path) as fh:
        raw = json.load(fh)
    return {gene_id: seq for gene_id, (seq, _) in raw.items()}


def _resolve_sequence(args) -> tuple[str, str]:
    """Return (target_id, sequence) from either --sequence or --gene-id."""
    if args.sequence:
        return "QUERY", args.sequence.strip().upper()

    sequences = _load_transcript_sequences(args.transcript_sequences)
    seq = sequences.get(args.gene_id)
    if not seq:
        raise SystemExit(
            f"Gene ID {args.gene_id!r} not found in {args.transcript_sequences} "
            f"({len(sequences):,} lncRNAs present). Pass --sequence to score a raw "
            "sequence instead."
        )
    return args.gene_id, seq.upper()


def score(
    sequence: str,
    cell_lines: list[str],
    model: xgb.XGBClassifier,
    vocab: list[str],
    include_distance: bool,
    distance: int | None = None,
    target: str = "QUERY",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (scores, X, columns) for one sequence across one or more cell lines.

    Builds features through build_lncrna_features() so this stays bit-identical to
    the training-time encoding. LncRnaRecord requires screen fields that are unknown
    at prediction time (day / rra_pvalue / fold_change / label) -- they are filled
    with placeholders and the returned label vector is discarded, since none of them
    feed a feature column.
    """
    k = len(vocab[0])
    records = [
        LncRnaRecord(
            target=target,
            cell_line=cell_line,
            day=14,
            rra_pvalue=1.0,
            fold_change=0.0,
            label=0,
            distance_to_closest_pc_gene=distance,
        )
        for cell_line in cell_lines
    ]

    X, _, columns = build_lncrna_features(
        records, {target: sequence}, k=k, include_distance=include_distance, vocab=vocab,
    )

    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != X.shape[1]:
        raise SystemExit(
            f"Feature width mismatch: model expects {expected} columns, built "
            f"{X.shape[1]} (k={k}, {len(vocab)} vocab k-mers, include_distance="
            f"{include_distance}). The vocab or --include-distance setting does not "
            "match the one used to train this model."
        )

    return model.predict_proba(X)[:, 1], X, columns


def main():
    parser = argparse.ArgumentParser(
        description="Score a lncRNA's essentiality (0-1) from its sequence + cell line.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gene-id", help="lncRNA gene_id to look up in --transcript-sequences.")
    src.add_argument("--sequence", help="Raw spliced transcript sequence (ACGT).")

    cell = parser.add_mutually_exclusive_group(required=True)
    cell.add_argument("--cell-line", choices=CELL_LINES, help="Cell line to score in.")
    cell.add_argument("--all-cell-lines", action="store_true",
                      help="Score the sequence in every cell line and print a table.")

    parser.add_argument("--model", default=_DEFAULT_MODEL,
                        help="Trained .ubj from scripts/train_lncrna_xgboost.py. "
                             "Vocab and params are read from the _vocab.json / "
                             "_params.json written alongside it.")
    parser.add_argument("--vocab", default=None, help="Override the derived vocab path.")
    parser.add_argument("--transcript-sequences", default=_DEFAULT_SEQUENCES,
                        help="{gene_id: [spliced_seq, \"\"]} JSON from lncfit.sequence "
                             "(--sequence-type transcript).")
    parser.add_argument("--distance", type=int, default=None,
                        help="distance_to_closest_pc_gene, only used if the model was "
                             "trained with it. Omitted -> -1, the training-time "
                             "convention for unknown.")
    parser.add_argument("--show-vector", action="store_true",
                        help="Print a summary of the built feature vector.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"No trained model at {model_path}. Train one first:\n"
            "  uv run python scripts/train_lncrna_xgboost.py --k 6 --output-dir data/model"
        )

    vocab_path = Path(args.vocab) if args.vocab else _sidecar(model_path, "vocab")
    if not vocab_path.exists():
        raise SystemExit(
            f"No vocab at {vocab_path}. It pins k-mer column order and cannot be "
            "refit here -- a rebuilt vocab would silently misalign every column."
        )
    with open(vocab_path) as fh:
        vocab = json.load(fh)
    if not vocab:
        raise SystemExit(f"Vocab at {vocab_path} is empty.")

    # include_distance must match training; read it from the params sidecar when present.
    params_path = _sidecar(model_path, "params")
    include_distance = False
    if params_path.exists():
        with open(params_path) as fh:
            include_distance = bool(json.load(fh).get("include_distance", False))
    elif args.distance is not None:
        include_distance = True

    target, sequence = _resolve_sequence(args)
    cell_lines = CELL_LINES if args.all_cell_lines else [args.cell_line]

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))

    scores, X, columns = score(
        sequence, cell_lines, model, vocab,
        include_distance=include_distance, distance=args.distance, target=target,
    )

    k = len(vocab[0])
    print(f"lncRNA           : {target} ({len(sequence):,} bp)")
    print(f"Model            : {model_path}")
    print(f"Features         : k={k}, {len(vocab):,} k-mers + {len(CELL_LINES)} cell one-hot"
          f"{' + distance' if include_distance else ''} = {len(columns):,} columns")

    if args.show_vector:
        row = X[0]
        nonzero = int(np.count_nonzero(row))
        print(f"Vector           : {nonzero:,} / {len(columns):,} non-zero, "
              f"k-mer freqs sum to {float(row[:len(vocab)].sum()):.4f}")

    print()
    if len(cell_lines) == 1:
        print(f"P(essential) in {cell_lines[0]}: {scores[0]:.4f}")
    else:
        width = max(len(c) for c in cell_lines)
        print(f"{'cell_line'.ljust(width)}  P(essential)")
        for cell_line, s in zip(cell_lines, scores):
            print(f"{cell_line.ljust(width)}  {s:.4f}")
    print("\nRanking score, not a calibrated probability (scale_pos_weight inflates it).")


if __name__ == "__main__":
    main()
