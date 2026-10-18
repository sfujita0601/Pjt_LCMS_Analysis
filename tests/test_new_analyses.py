"""moderated t 検定・フォレストプロット用の効果量・飼料組成との比較の検証 (模擬データ)。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcms_diet as ld  # noqa: E402
import lcms_stats as ls  # noqa: E402


# ---------------------------------------------------------------- moderated t
def _simulate(G=2000, n=3, d0=6.0, s02=0.2, effect=None, seed=1):
    rng = np.random.default_rng(seed)
    sig2 = d0 * s02 / rng.chisquare(d0, G)
    eff = np.zeros(G) if effect is None else effect
    A = [rng.normal(e, np.sqrt(v), n) for e, v in zip(eff, sig2)]
    B = [rng.normal(0, np.sqrt(v), n) for v in sig2]
    return A, B


def test_prior_is_recovered():
    A, B = _simulate()
    r = ls.moderated_t(A, B)
    assert r["d0"] == pytest.approx(6.0, rel=0.25)
    assert r["s02"] == pytest.approx(0.2, rel=0.15)


def test_type_one_error_is_nominal():
    A, B = _simulate(seed=2)
    r = ls.moderated_t(A, B)
    assert 0.03 <= np.mean(r["p"] < 0.05) <= 0.07


def test_moderated_t_more_powerful_than_welch():
    from scipy import stats

    eff = np.zeros(2000)
    eff[:300] = 1.0
    A, B = _simulate(effect=eff, seed=3)
    r = ls.moderated_t(A, B)
    pw = np.array([stats.ttest_ind(a, b, equal_var=False).pvalue for a, b in zip(A[:300], B[:300])])
    assert np.mean(r["p"][:300] < 0.05) > np.mean(pw < 0.05) + 0.1


def test_trigamma_inverse():
    from scipy.special import polygamma

    for y in [0.01, 0.5, 2.0, 10.0]:
        assert polygamma(1, ls._trigamma_inverse(y)) == pytest.approx(y, rel=1e-6)


def test_compare_groups_moderated_columns():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        cond = "T" if i < 3 else "C"
        rows.append({"condition": cond, **{f"m{j}": float(2 ** (rng.normal(1.0 if (cond == "T" and j == 0) else 0, 0.3)))
                                          for j in range(30)}})
    df = pd.DataFrame(rows)
    res = ls.compare_groups(df, [f"m{j}" for j in range(30)], "T", "C", ls.MODT, "Benjamini-Hochberg (FDR)")
    assert {"moderated t", "自由度 (d + d0)", "95% CI 下限", "95% CI 上限"} <= set(res.columns)
    assert "d0" in res.attrs["prior"]
    r0 = res.set_index("化合物").loc["m0"]
    assert r0["95% CI 下限"] < r0["log2 幾何平均比"] < r0["95% CI 上限"]
    assert r0["log2 幾何平均比"] > 0.5


# ---------------------------------------------------------------- 飼料組成
def test_diet_column_matching():
    compounds = ["Glutamic acid", "Ornitine", "Leucine", "Cystine", "Cysteine", "Valine"]
    m = ld.match_compounds(["Glu", "Orn", "ロイシン", "Cys", "Val (g/kg)", "Xyz"], compounds)
    assert m == {"Glu": "Glutamic acid", "Orn": "Ornitine", "ロイシン": "Leucine", "Cys": "Cysteine",
                 "Val (g/kg)": "Valine", "Xyz": ""}


def test_diet_ratio_and_classification():
    diet = pd.DataFrame({"Leu": [16.0, 8.0], "Gly": [3.5, 3.5]}, index=["Ctrl", "Low"])
    ratio = ld.diet_log2_ratio(diet, {"C": "Ctrl", "L": "Low"}, "C")
    assert ratio.at["L", "Leu"] == pytest.approx(-1.0) and ratio.at["L", "Gly"] == pytest.approx(0.0)
    plasma = pd.DataFrame({"比較群": ["L", "L"], "化合物": ["Leucine", "Glycine"], "log2 幾何平均比": [-0.8, 0.9],
                           "95% CI 下限": [-1.2, 0.5], "95% CI 上限": [-0.4, 1.3], "p": [0.001, 0.002],
                           "q": [0.01, 0.01]})
    t = ld.combine(ratio, plasma, {"Leu": "Leucine", "Gly": "Glycine"}, q_thr=0.05, diet_thr=0.25).set_index("化合物")
    assert t.at["Leucine", "分類"] == "飼料と同じ向きに有意に変化"
    assert t.at["Glycine", "分類"] == "飼料はほぼ同じだが血中が有意に変化"


def test_diet_ratio_zero_control_is_nan():
    diet = pd.DataFrame({"Arg": [0.0, 5.0]}, index=["Ctrl", "High"])
    ratio = ld.diet_log2_ratio(diet, {"C": "Ctrl", "H": "High"}, "C")
    assert np.isnan(ratio.at["H", "Arg"])
