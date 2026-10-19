"""レーダーチャートの評価値とばらつき (誤差の伝搬則) の検証。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcms_radar as lr  # noqa: E402


def test_fc_error_propagation_matches_simulation():
    # 平均 10 (SD 1) と 5 (SD 0.5) の比: 伝搬則の SD(FC) = 2 * √(0.1² + 0.1²) ≈ 0.283
    a, b = np.array([9.0, 10.0, 11.0]), np.array([4.5, 5.0, 5.5])
    est, lo, hi, _ = lr.estimate(a, b, lr.M_FC, disp=lr.D_SD)
    assert est == pytest.approx(2.0)
    assert (hi - est) == pytest.approx(2 * np.sqrt(0.1 ** 2 + 0.1 ** 2))
    rng = np.random.default_rng(0)
    sim = rng.normal(10, 1, 200000) / rng.normal(5, 0.5, 200000)
    assert np.std(sim) == pytest.approx(hi - est, rel=0.06)


def test_sem_and_ci_are_narrower_and_wider():
    a, b = np.array([9.0, 10.0, 11.0, 12.0]), np.array([4.5, 5.0, 5.5, 6.0])
    sd = lr.estimate(a, b, lr.M_DIFF, disp=lr.D_SD)
    sem = lr.estimate(a, b, lr.M_DIFF, disp=lr.D_SEM)
    ci = lr.estimate(a, b, lr.M_DIFF, disp=lr.D_CI)
    assert sd[0] == sem[0] == ci[0] == pytest.approx(5.25)
    assert (sem[2] - sem[0]) == pytest.approx((sd[2] - sd[0]) / 2)
    assert (ci[2] - ci[0]) > (sem[2] - sem[0])


def test_log2_fc_is_geometric_ratio_with_asymmetric_range():
    a, b = np.log2([8.0, 16.0, 32.0]), np.log2([2.0, 4.0, 8.0])
    est, lo, hi, _ = lr.estimate(a, b, lr.M_FC, disp=lr.D_SD, log2=True)
    assert est == pytest.approx(4.0)
    assert est / lo == pytest.approx(hi / est)  # 対数の上で対称
    d, dlo, dhi, _ = lr.estimate(a, b, lr.M_LOG2FC, disp=lr.D_SD, log2=True)
    assert d == pytest.approx(2.0) and dhi - d == pytest.approx(np.sqrt(2))


def test_reference_self_has_only_own_spread():
    b = np.array([4.0, 5.0, 6.0])
    est, lo, hi, _ = lr.estimate(b, b, lr.M_FC, disp=lr.D_SD, ref_self=True)
    assert est == pytest.approx(1.0) and hi - est == pytest.approx(1 / 5)


def test_t_hedges_z():
    a, b = np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0])
    assert lr.estimate(a, b, lr.M_T)[0] == pytest.approx(2 / np.sqrt(2 / 3))
    g = lr.estimate(a, b, lr.M_G)[0]
    assert g == pytest.approx(2 * (1 - 3 / (4 * 6 - 9)))
    assert lr.estimate(a, b, lr.M_Z, disp=lr.D_SD)[:3] == pytest.approx((2.0, 1.0, 3.0))


def test_median_uses_robust_spread():
    a = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    est, lo, hi, note = lr.estimate(a, a, lr.M_VALUE, center=lr.C_MEDIAN, disp=lr.D_SD)
    assert (est, lo, hi) == (3.0, 2.0, 4.0) and note == "四分位範囲"


def test_radar_table_and_bootstrap():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"condition": ["C"] * 6 + ["T"] * 6,
                       "x": np.r_[rng.normal(10, 1, 6), rng.normal(20, 2, 6)],
                       "y": np.r_[rng.normal(5, 1, 6), rng.normal(5, 1, 6)]})
    t = lr.radar_table(df, ["x", "y"], "C", ["T"], lr.M_FC, disp=lr.D_BOOT, n_boot=500, include_ref=True)
    assert list(t["condition"]) == ["C", "C", "T", "T"]
    tx = t[(t["condition"] == "T") & (t["化合物"] == "x")].iloc[0]
    assert tx["下限"] < tx["値"] < tx["上限"] and tx["値"] == pytest.approx(2.0, rel=0.2)
    s = lr.rescale(lr.radar_table(df, ["x", "y"], "C", ["C", "T"], lr.M_VALUE), lr.S_MAX)
    assert s.groupby("化合物")["値"].max().tolist() == pytest.approx([1.0, 1.0])
