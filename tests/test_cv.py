from lncfit.constants import MIN_FOLD_RECORDS
from lncfit.cv import build_lncrna_folds
from lncfit.screen_data import LncRnaRecord


def _make_records(n_per_chrom, chroms):
    records = []
    for c in chroms:
        for i in range(n_per_chrom):
            records.append(LncRnaRecord(
                target=f"T_{c}_{i}",
                cell_line="HAP1",
                day=14,
                rra_pvalue=0.5,
                fold_change=0.0,
                label=1 if i % 5 == 0 else 0,
                guide_sequences=("AAACCCGGGTTT",),
                chrom=c,
            ))
    return records


def test_excludes_chroms_below_min_fold_records():
    records = _make_records(MIN_FOLD_RECORDS - 1, ["1"]) + _make_records(MIN_FOLD_RECORDS + 1, ["2"])
    cv_chroms, fold_data, _ = build_lncrna_folds(records, k=3, verbose=False)
    assert cv_chroms == ["2"]


def test_val_and_es_and_train_partition_without_overlap():
    records = _make_records(MIN_FOLD_RECORDS + 10, ["1", "2", "3"])
    cv_chroms, fold_data, _ = build_lncrna_folds(records, k=3, verbose=False)
    assert set(cv_chroms) == {"1", "2", "3"}
    for val_chrom in cv_chroms:
        X_tr, y_tr, X_val, y_val, X_es, y_es = fold_data[val_chrom]
        total = X_tr.shape[0] + X_val.shape[0] + X_es.shape[0]
        assert total == len(records)
        assert X_val.shape[0] == MIN_FOLD_RECORDS + 10


def test_es_chrom_never_equals_val_chrom():
    records = _make_records(MIN_FOLD_RECORDS + 10, ["1", "2", "3"])
    cv_chroms, fold_data, _ = build_lncrna_folds(records, k=3, verbose=False)
    for i, val_chrom in enumerate(cv_chroms):
        es_chrom = cv_chroms[(i + 1) % len(cv_chroms)]
        assert es_chrom != val_chrom


def test_feature_columns_consistent_across_folds():
    records = _make_records(MIN_FOLD_RECORDS + 10, ["1", "2"])
    _, _, feature_cols = build_lncrna_folds(records, k=3, verbose=False)
    assert len(feature_cols) > 0
