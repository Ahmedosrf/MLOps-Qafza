"""Check error accounting and train-defined distribution comparisons."""

import numpy as np
import pandas as pd
import pytest
from Task2.src.diagnose_model import grouped_errors, numeric_psi


def test_grouped_errors_accounts_for_false_positives_and_missed_delays():
    frame = pd.DataFrame(
        {
            "state": ["SP"] * 4 + ["RJ"],
            "is_late": [1, 1, 0, 0, 1],
            "predicted_late": [1, 0, 1, 0, 0],
        }
    )
    rows = grouped_errors(frame, "state", min_rows=2)
    assert len(rows) == 1
    assert rows[0]["false_positives"] == 1
    assert rows[0]["false_negatives"] == 1
    assert rows[0]["precision"] == 0.5
    assert rows[0]["recall"] == 0.5


def test_psi_uses_reference_bins_and_detects_out_of_range_distribution():
    reference = np.arange(100, dtype=float)
    assert numeric_psi(reference, reference) == pytest.approx(0.0)
    assert numeric_psi(reference, reference + 200.0) > 0.0
    assert numeric_psi(np.ones(10), np.ones(10)) == 0.0
