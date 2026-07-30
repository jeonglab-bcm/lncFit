#!/usr/bin/env python3
"""Validate lncRNA-classifier leaderboard submissions and (re)build LEADERBOARD.md.

A submission is a directory under results/<challenge>/leaderboard/submissions/<name>/
containing:
  predictions.csv   target, cell_line, y_pred_proba for every held-out test row
                     (exactly the format lncfit.pipeline.LncRnaPipeline writes to
                     a run's predictions.csv -- just copy that file over)
  submission.yaml    {submitter: "...", model: "...", description: "..." (optional)}

AUROC/AUPRC are recomputed here directly from predictions.csv against the real
held-out labels in the test set -- a submitted metrics.csv or y_true column is
never trusted for scoring, only target/cell_line/y_pred_proba are read.

Each challenge declares its own held-out labeled set via
results/<challenge>/leaderboard/challenge.yaml:
  test_path: data/processed/....jsonl.gz
  title: optional one-line description for the LEADERBOARD.md header
  exclude_cell_lines: optional list, e.g. [HEK293FT] -- dropped from scoring
    entirely. A submission's predictions.csv may still include rows for an
    excluded cell line (harmless, ignored); it just isn't required to.
  only_cell_lines: optional list, e.g. [THP1] -- score ONLY these cell lines.
    Lets a single-held-out-cell-line challenge reuse the full dataset as its
    ground truth instead of shipping a separate answers file. Applied after
    exclude_cell_lines.

Usage:
  python scripts/build_leaderboard.py --challenge lncrna_rra_day14
  python scripts/build_leaderboard.py --challenge lncrna_rra_day14_cellline_loco

Exits non-zero (after still writing LEADERBOARD.md for whatever validated) if
any submission is malformed, missing, or doesn't cover the held-out test set
exactly -- so CI can fail the check without silently dropping a bad entry.
"""
import argparse
import html
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from lncfit.screen_data import LncRnaRecord, load_jsonl
from lncfit.xgboost_model import evaluate_lncrna_by_group

_REQUIRED_PRED_COLS = {"target", "cell_line", "y_pred_proba"}
# uses_measured_depletion is required so compliance with the no-measured-depletion
# rule is an explicit assertion by the submitter rather than an unstated assumption.
# We cannot inspect someone's feature matrix, so the board asks them to declare it;
# that is the same honour system the rest of the scoring rests on, but at least it
# is on the record and forces the question to be answered.
_REQUIRED_SUBMISSION_FIELDS = {"submitter", "model", "uses_measured_depletion"}
# GitHub username rules: alphanumeric or single hyphens, no leading/trailing/
# double hyphens, max 39 chars.
_GITHUB_HANDLE_RE = re.compile(r"^[A-Za-z\d](?:[A-Za-z\d]|-(?=[A-Za-z\d])){0,38}$")


def _repo_slug() -> str:
    """Best-effort 'owner/repo' from the origin remote, for linking off the
    GitHub Pages site (which only serves docs/, not the rest of the repo)."""
    try:
        url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return ""
    url = url.removesuffix(".git")
    for prefix in ("git@github.com:", "https://github.com/"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return ""


class SubmissionError(Exception):
    pass


# 2000 resamples puts the interval endpoints within ~0.001, far finer than the
# interval's own width (~0.12 here). Fixed seed because CI regenerates this file on
# every submission and an interval that jittered run-to-run would read as a scoring
# change: same predictions must always give the same published interval.
_BOOTSTRAP_N = 2000
_BOOTSTRAP_SEED = 0


def _bootstrap_auprc_ci(y_true, y_pred_proba, n: int = _BOOTSTRAP_N) -> tuple[float, float]:
    """95% percentile bootstrap interval for AUPRC, resampling rows.

    Published next to the point estimate because with 202 positives the point
    estimate alone implies a precision the data does not support -- the interval is
    ~0.12 wide, so four-decimal ranks invite people to chase differences that are
    pure sampling noise. Resamples that happen to contain no positives are skipped
    (AUPRC is undefined there).
    """
    y = np.asarray(y_true)
    p = np.asarray(y_pred_proba, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y[idx], p[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def _load_truth(test_path: str, exclude_cell_lines: set[str] | None = None,
                only_cell_lines: set[str] | None = None):
    """Returns (scored_records, truth, excluded_keys).

    exclude_cell_lines are dropped from scoring entirely (e.g. HEK293FT, which
    has no Celligner data and isn't a real cancer cell line -- excluded from
    both leaderboard challenges). excluded_keys is still returned so
    _score_submission can tell "a row for an excluded cell line" (tolerated,
    just ignored) apart from "a row that shouldn't exist at all" (a real error).
    """
    exclude_cell_lines = exclude_cell_lines or set()
    all_records = load_jsonl(test_path, record_cls=LncRnaRecord)

    def _scored(r) -> bool:
        if r.cell_line in exclude_cell_lines:
            return False
        return not only_cell_lines or r.cell_line in only_cell_lines

    records = [r for r in all_records if _scored(r)]
    if not records:
        raise ValueError(
            f"no rows left to score from {test_path} after applying "
            f"exclude_cell_lines={sorted(exclude_cell_lines)} / "
            f"only_cell_lines={sorted(only_cell_lines or [])}"
        )
    truth = {(r.target, r.cell_line): r.label for r in records}
    excluded_keys = {(r.target, r.cell_line) for r in all_records if not _scored(r)}
    return records, truth, excluded_keys


def _score_submission(sub_dir: Path, records: list, truth: dict, excluded_keys: set) -> dict:
    submission_yaml = sub_dir / "submission.yaml"
    predictions_csv = sub_dir / "predictions.csv"

    if not submission_yaml.exists():
        raise SubmissionError(f"{sub_dir.name}: missing submission.yaml")
    if not predictions_csv.exists():
        raise SubmissionError(f"{sub_dir.name}: missing predictions.csv")

    with open(submission_yaml) as fh:
        meta = yaml.safe_load(fh) or {}
    missing_fields = _REQUIRED_SUBMISSION_FIELDS - meta.keys()
    if missing_fields:
        raise SubmissionError(
            f"{sub_dir.name}: submission.yaml missing field(s): {sorted(missing_fields)}"
        )

    uses_depletion = meta["uses_measured_depletion"]
    if not isinstance(uses_depletion, bool):
        raise SubmissionError(
            f"{sub_dir.name}: uses_measured_depletion must be true or false, got "
            f"{uses_depletion!r} -- declare whether any measured fold_change / "
            "rra_pvalue / guide depletion (from ANY cell line) fed your features. "
            "See docs/PARTICIPATE.md."
        )

    submitter = str(meta["submitter"]).strip()
    if not _GITHUB_HANDLE_RE.match(submitter):
        raise SubmissionError(
            f"{sub_dir.name}: submitter {submitter!r} doesn't look like a GitHub handle "
            "(letters/digits/single-hyphens, <=39 chars, no leading/trailing hyphen) -- "
            "use your (or your team's) actual GitHub username so it can be linked and verified"
        )

    preds = pd.read_csv(predictions_csv)
    missing_cols = _REQUIRED_PRED_COLS - set(preds.columns)
    if missing_cols:
        raise SubmissionError(
            f"{sub_dir.name}: predictions.csv missing column(s): {sorted(missing_cols)}"
        )

    if preds.duplicated(subset=["target", "cell_line"]).any():
        raise SubmissionError(f"{sub_dir.name}: predictions.csv has duplicate (target, cell_line) rows")

    pred_keys = set(zip(preds["target"], preds["cell_line"]))
    truth_keys = set(truth.keys())
    missing_rows = truth_keys - pred_keys
    # Rows for an excluded cell line (e.g. HEK293FT) are tolerated, not required --
    # most submitters will just copy predictions.csv straight from a pipeline run
    # that scored all 5 cell lines, so don't punish them for including it.
    extra_rows = pred_keys - truth_keys - excluded_keys
    if missing_rows:
        raise SubmissionError(
            f"{sub_dir.name}: predictions.csv is missing {len(missing_rows)} row(s) "
            "required by the held-out test set"
        )
    if extra_rows:
        raise SubmissionError(
            f"{sub_dir.name}: predictions.csv has {len(extra_rows)} row(s) not in the "
            "held-out test set"
        )

    pred_lookup = {(row.target, row.cell_line): row.y_pred_proba for row in preds.itertuples()}
    y_true = [truth[(r.target, r.cell_line)] for r in records]
    y_pred_proba = [pred_lookup[(r.target, r.cell_line)] for r in records]

    metrics_rows = evaluate_lncrna_by_group(records, y_true, y_pred_proba)
    overall = next(r for r in metrics_rows if r["split"] == "Overall")

    return {
        "name": sub_dir.name,
        "submitter": submitter,
        "model": str(meta["model"]),
        "description": str(meta.get("description", "")).strip(),
        "auroc": overall["auroc"],
        "auprc": overall["auprc"],
        "auprc_ci": _bootstrap_auprc_ci(y_true, y_pred_proba),
        "metrics_rows": metrics_rows,
        "has_config": (sub_dir / "config.yaml").exists(),
        # Scored either way, but ranked separately -- see _render_leaderboard.
        "ineligible": bool(uses_depletion),
    }


def _render_leaderboard(challenge: str, title: str, rows: list[dict], n_errors: int) -> str:
    lines = [
        f"# Leaderboard -- {challenge}",
        "",
    ]
    if title:
        lines += [title, ""]
    lines += [
        "Auto-generated by `scripts/build_leaderboard.py` (via CI) -- do not edit by hand.",
        "AUROC/AUPRC are recomputed directly from each submission's `predictions.csv` "
        "against the real held-out test labels, not read from any submitted metrics file.",
        "",
        "**AUPRC carries a 95% bootstrap CI about 0.12 wide here (202 positives). "
        "Gaps smaller than the intervals' overlap are noise, not progress.**",
        "",
        "**[How to enter](../../../docs/PARTICIPATE.md)**",
        "",
        "| Rank | Submitter | Model | AUPRC | 95% CI | AUROC | Submission |",
        "|---|---|---|---|---|---|---|",
    ]
    eligible = [r for r in rows if not r["ineligible"]]
    ineligible = [r for r in rows if r["ineligible"]]
    for i, r in enumerate(eligible, 1):
        lo, hi = r["auprc_ci"]
        lines.append(
            f"| {i} | [@{r['submitter']}](https://github.com/{r['submitter']}) | {r['model']} | "
            f"{r['auprc']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['auroc']:.4f} | "
            f"[{r['name']}](submissions/{r['name']}/) |"
        )
    if ineligible:
        lines += [
            "",
            "## Ineligible -- used measured depletion as a feature",
            "",
            "These declared `uses_measured_depletion: true`, so they are scored but not "
            "ranked: they use measured `fold_change` / `rra_pvalue` / guide depletion from "
            "the training cell lines as *input features*, which "
            "[the rules](../../../docs/PARTICIPATE.md#no-measured-depletion-as-a-feature--any-cell-line-any-day) "
            "no longer permit. That shortcut predicts pan-essentiality without needing to "
            "understand sequence at all, so it does not answer the question this challenge "
            "asks.",
            "",
            "Kept visible, with scores, because each declared its features openly and "
            "complied with the rules as written when it was submitted. This is a rule "
            "change, not a finding of misconduct.",
            "",
            "| Submitter | Model | AUPRC | 95% CI | AUROC | Submission |",
            "|---|---|---|---|---|---|",
        ]
        for r in ineligible:
            lo, hi = r["auprc_ci"]
            lines.append(
                f"| [@{r['submitter']}](https://github.com/{r['submitter']}) | {r['model']} | "
                f"{r['auprc']:.4f} | [{lo:.4f}, {hi:.4f}] | {r['auroc']:.4f} | "
                f"[{r['name']}](submissions/{r['name']}/) |"
            )
    if n_errors:
        lines += ["", f"{n_errors} submission(s) failed validation and are not scored above -- see CI log."]
    return "\n".join(lines) + "\n"


_PAGE_STYLE = """
:root { color-scheme: light dark; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  max-width: 960px; margin: 2rem auto; padding: 0 1rem;
  line-height: 1.5;
}
h1 { font-size: 1.5rem; }
.subtitle { color: #666; margin-top: -0.5rem; }
@media (prefers-color-scheme: dark) { .subtitle { color: #aaa; } }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #ddd; }
@media (prefers-color-scheme: dark) { th, td { border-bottom: 1px solid #444; } }
th { font-weight: 600; }
tr.rank-1 td:first-child { font-weight: 700; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
details { margin: 0; }
summary { cursor: pointer; color: #0969da; }
@media (prefers-color-scheme: dark) { summary { color: #58a6ff; } }
.detail-table { margin: 0.5rem 0 1rem 0; }
.detail-table th, .detail-table td { padding: 0.25rem 0.6rem; border-bottom: none; }
.errors { color: #cf222e; }
@media (prefers-color-scheme: dark) { .errors { color: #ff7b72; } }
footer { color: #666; font-size: 0.85rem; margin-top: 2rem; }
@media (prefers-color-scheme: dark) { footer { color: #aaa; } }
a { color: inherit; }
"""


def _participate_link(repo: str) -> str:
    """Link the how-to-enter guide from both pages -- someone who lands on the board
    from outside the repo has no other route to it. Points at the GitHub blob rather
    than docs/PARTICIPATE.md, which Pages serves as raw text."""
    if not repo:
        return ""
    url = f"https://github.com/{html.escape(repo)}/blob/main/docs/PARTICIPATE.md"
    return f'<p><strong><a href="{url}">How to enter &rarr;</a></strong></p>'


def _render_leaderboard_html(challenge: str, title: str, rows: list[dict], n_errors: int, repo: str) -> str:
    def esc(s) -> str:
        return html.escape(str(s))

    tree_base = f"https://github.com/{repo}/tree/main/results/{challenge}/leaderboard/submissions" if repo else None
    blob_base = f"https://github.com/{repo}/blob/main/results/{challenge}/leaderboard/submissions" if repo else None

    def _row(r: dict, rank) -> str:
        sub_link = f"{tree_base}/{esc(r['name'])}/" if tree_base else f"submissions/{esc(r['name'])}/"
        config_url = f"{blob_base}/{esc(r['name'])}/config.yaml" if blob_base else f"submissions/{esc(r['name'])}/config.yaml"
        detail_rows = "".join(
            f"<tr><td>{esc(m['split'])}</td><td class='num'>{m['n']:,}</td>"
            f"<td class='num'>{m['auroc']:.4f}</td><td class='num'>{m['auprc']:.4f}</td>"
            f"<td class='num'>{m['f1']:.4f}</td></tr>"
            for m in r["metrics_rows"]
        )
        config_link = (
            f" &middot; <a href='{config_url}'>config.yaml</a>" if r.get("has_config") else ""
        )
        lo, hi = r["auprc_ci"]
        return f"""
<tr class="rank-{rank if rank else 'x'}">
  <td>{rank if rank else '&mdash;'}</td>
  <td><a href="https://github.com/{esc(r['submitter'])}">@{esc(r['submitter'])}</a></td>
  <td>{esc(r['model'])}</td>
  <td class="num">{r['auprc']:.4f}</td>
  <td class="num">[{lo:.4f}, {hi:.4f}]</td>
  <td class="num">{r['auroc']:.4f}</td>
  <td><details><summary>details</summary>
    {f"<p>{esc(r['description'])}</p>" if r['description'] else ""}
    <table class="detail-table">
      <tr><th>Split</th><th class="num">n</th><th class="num">AUROC</th><th class="num">AUPRC</th><th class="num">F1</th></tr>
      {detail_rows}
    </table>
    <p><a href="{sub_link}">submission folder</a>{config_link}</p>
  </details></td>
</tr>"""

    body_rows = [_row(r, i) for i, r in enumerate(
        (r for r in rows if not r["ineligible"]), 1)]
    ineligible_rows = [_row(r, None) for r in rows if r["ineligible"]]
    ineligible_html = f"""
<h2>Ineligible &mdash; used measured depletion as a feature</h2>
<p>Scored but not ranked: these declared <code>uses_measured_depletion: true</code>,
using measured fold-change / p-value / guide depletion from the training cell lines as
input features. That predicts pan-essentiality without needing to understand sequence,
so it does not answer this challenge's question. Kept visible, with scores, because each
declared its features openly and complied with the rules as written at submission time
&mdash; this is a rule change, not a finding of misconduct.</p>
<table>
<tr><th>Rank</th><th>Submitter</th><th>Model</th><th class="num">AUPRC</th><th class="num">95% CI</th><th class="num">AUROC</th><th></th></tr>
{"".join(ineligible_rows)}
</table>""" if ineligible_rows else ""

    errors_html = (
        f'<p class="errors">{n_errors} submission(s) failed validation and are not shown above -- see CI logs.</p>'
        if n_errors else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(challenge)} leaderboard</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<p><a href="index.html">&larr; all challenges</a></p>
<h1>{esc(challenge)}</h1>
{f'<p class="subtitle">{esc(title)}</p>' if title else ""}
{_participate_link(repo)}
<p><strong>AUPRC carries a 95% bootstrap CI about 0.12 wide here (202 positives).
Gaps smaller than the intervals' overlap are noise, not progress.</strong></p>
<table>
<tr><th>Rank</th><th>Submitter</th><th>Model</th><th class="num">AUPRC</th><th class="num">95% CI</th><th class="num">AUROC</th><th></th></tr>
{"".join(body_rows) if body_rows else '<tr><td colspan="7">No eligible submissions yet.</td></tr>'}
</table>
{ineligible_html}
{errors_html}
<footer>Auto-generated by <code>scripts/build_leaderboard.py</code> via CI on every submission PR. AUROC/AUPRC are recomputed independently from each submission's predictions.csv against the real held-out labels.</footer>
</body>
</html>
"""


def _render_index_html(challenges: list[dict], repo: str) -> str:
    def _item(c: dict) -> str:
        slug = html.escape(c["slug"])
        title_html = f' &mdash; {html.escape(c["title"])}' if c["title"] else ""
        return (
            f'<li><a href="{slug}.html">{slug}</a>{title_html} '
            f'<span class="count">({c["n_valid"]} submission(s))</span></li>'
        )

    items = "".join(_item(c) for c in challenges)
    repo_link = f'<p><a href="https://github.com/{html.escape(repo)}">{html.escape(repo)}</a> on GitHub.</p>' if repo else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>lncFit leaderboards</title>
<style>{_PAGE_STYLE}
.count {{ color: #666; }}
@media (prefers-color-scheme: dark) {{ .count {{ color: #aaa; }} }}
</style>
</head>
<body>
<h1>lncFit leaderboards</h1>
<ul>{items if items else "<li>No challenges yet.</li>"}</ul>
{_participate_link(repo)}
{repo_link}
<footer>Auto-generated by <code>scripts/build_leaderboard.py</code> via CI.</footer>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--challenge", default="lncrna_rra_day14", help="results/<challenge>/leaderboard/...")
    parser.add_argument(
        "--test-path", default=None,
        help="Held-out labeled test set. Overrides challenge.yaml's test_path if given.",
    )
    args = parser.parse_args()

    challenge_dir = Path("results") / args.challenge / "leaderboard"
    submissions_dir = challenge_dir / "submissions"
    leaderboard_path = challenge_dir / "LEADERBOARD.md"
    challenge_config_path = challenge_dir / "challenge.yaml"

    if not submissions_dir.exists():
        sys.exit(f"No submissions directory found at {submissions_dir}")

    test_path = args.test_path
    title = ""
    exclude_cell_lines = set()
    only_cell_lines = set()
    if challenge_config_path.exists():
        with open(challenge_config_path) as fh:
            challenge_config = yaml.safe_load(fh) or {}
        test_path = test_path or challenge_config.get("test_path")
        title = challenge_config.get("title", "").strip()
        exclude_cell_lines = set(challenge_config.get("exclude_cell_lines") or [])
        only_cell_lines = set(challenge_config.get("only_cell_lines") or [])
    if not test_path:
        sys.exit(
            f"No test set specified -- pass --test-path or add test_path to {challenge_config_path}"
        )

    records, truth, excluded_keys = _load_truth(test_path, exclude_cell_lines, only_cell_lines)

    rows = []
    errors = []
    for sub_dir in sorted(submissions_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        try:
            rows.append(_score_submission(sub_dir, records, truth, excluded_keys))
        except SubmissionError as e:
            errors.append(str(e))

    rows.sort(key=lambda r: r["auprc"], reverse=True)

    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(_render_leaderboard(args.challenge, title, rows, len(errors)))
    print(f"Wrote {leaderboard_path} ({len(rows)} valid submission(s), {len(errors)} error(s))")

    repo = _repo_slug()
    docs_dir = Path("docs")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"{args.challenge}.html").write_text(
        _render_leaderboard_html(args.challenge, title, rows, len(errors), repo)
    )

    # Rebuild the shared index across every known challenge, not just this one,
    # so it stays consistent regardless of which challenge's submissions changed.
    challenges = []
    for other_config in sorted(Path("results").glob("*/leaderboard/challenge.yaml")):
        slug = other_config.parent.parent.name
        with open(other_config) as fh:
            other_cfg = yaml.safe_load(fh) or {}
        if slug == args.challenge:
            n_valid = len(rows)
        else:
            other_leaderboard = other_config.parent / "LEADERBOARD.md"
            n_valid = other_leaderboard.read_text().count("](submissions/") if other_leaderboard.exists() else 0
        challenges.append({"slug": slug, "title": other_cfg.get("title", ""), "n_valid": n_valid})
    (docs_dir / "index.html").write_text(_render_index_html(challenges, repo))
    print(f"Wrote {docs_dir / f'{args.challenge}.html'} and {docs_dir / 'index.html'}")

    if errors:
        print("\nValidation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
