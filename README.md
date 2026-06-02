# lncFit

lncRNA fitness predictor using [ChatNT](https://huggingface.co/InstaDeepAI/ChatNT).

> **Important:** lncRNA essentiality / CRISPR-screen log2FC prediction is **not** listed among
> ChatNT's documented training tasks (see [Datasets_overview.csv](https://huggingface.co/InstaDeepAI/ChatNT/blob/main/Datasets_overview.csv)).
> All outputs are out-of-distribution exploratory estimates and must be treated as such until
> benchmarked against known CRISPR-screen labels.

## Installation

```bash
pip install -r requirements.txt
```

The ChatNT model is approximately 8B parameters. A GPU-capable environment is strongly recommended.
The model uses `trust_remote_code=True` and is licensed for **non-commercial use only**.

## Usage

### Single sequence

```bash
python scripts/run_chatnt.py --cell-line K562 --dna-sequence ACGTACGT
```

### Multiple sequences

```bash
python scripts/run_chatnt.py \
  --cell-line K562 \
  --dna-sequence ACGTACGT \
  --dna-sequence TGCATGCA
```

### FASTA input

```bash
python scripts/run_chatnt.py \
  --cell-line K562 \
  --fasta examples/lncrna_regions.fa
```

### Dry run (no model download)

```bash
python scripts/run_chatnt.py \
  --cell-line K562 \
  --dna-sequence ACGTACGT \
  --dry-run
```

The `--dry-run` flag prints the constructed prompt and exits without loading ChatNT.

## Running tests

```bash
pytest tests/
```

Tests do not download model weights.
