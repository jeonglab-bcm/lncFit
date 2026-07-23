"""Run the pluggable, YAML-configured lncRNA classifier pipeline (issue #78 follow-up).

One config file, one command, every axis is a choice instead of a script:
model (lncfit.classifiers registry), features (k-mer vs DNABERT-2), cell-line
embedding (one-hot vs Celligner UMAP/PCA), tuning (fixed/grid/optuna), and
cross-validation (none/chromosome/stratified).

See configs/README.md for the config schema and configs/pipeline/*.yaml for
ready-to-run examples.

Usage:
  uv run python scripts/run_pipeline.py --config configs/pipeline/xgboost_kmer_fixed.yaml
  uv run python scripts/run_pipeline.py --config configs/pipeline/xgboost_kmer_optuna.yaml
  uv run python scripts/run_pipeline.py --config configs/pipeline/logreg_kmer_grid.yaml \\
      --output-dir /tmp/my_run
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.pipeline import LncRnaPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to a pipeline YAML config (see configs/README.md).")
    parser.add_argument("--output-dir", default=None, help="Override the config's output_dir.")
    args = parser.parse_args()

    pipeline = LncRnaPipeline.from_yaml(args.config)
    if args.output_dir:
        pipeline.output_dir = Path(args.output_dir)

    pipeline.run()


if __name__ == "__main__":
    main()
