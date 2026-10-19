"""IS 補正グループ: 同じバッチのマウスとラットを別々の IS 平均で補正する。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcms_pipeline as lp  # noqa: E402
from make_conc_table import DILUTION_COL, FILE_COL, LABEL_COL  # noqa: E402

IS = "MES"


def _raw():
    labels = ["M1", "M2", "R1", "R2", "STD_1"]
    return pd.DataFrame({FILE_COL: [f"f{i}" for i in range(5)], LABEL_COL: labels, DILUTION_COL: [""] * 5,
                         IS: [100.0, 120.0, 300.0, 340.0, 200.0]})


def test_without_groups_uses_common_mean():
    f = lp.is_factors(_raw(), IS)
    assert f["f0"] == pytest.approx(215 / 100)


def test_groups_use_their_own_mean():
    groups = {"M1": "マウス", "M2": "マウス", "R1": "ラット", "R2": "ラット"}
    f = lp.is_factors(_raw(), IS, groups=groups)
    assert f["f0"] == pytest.approx(110 / 100) and f["f1"] == pytest.approx(110 / 120)
    assert f["f2"] == pytest.approx(320 / 300) and f["f3"] == pytest.approx(320 / 340)
    assert f["f4"] == pytest.approx(215 / 200)  # STD は希釈グループ全体の平均
    m = lp.is_means(_raw(), IS, groups=groups).set_index(lp.IS_GROUP_COL)
    assert m.loc["マウス", "平均"] == pytest.approx(110) and m.loc["ラット", "n"] == 2


def test_unused_samples_and_empty_group():
    groups = {"R1": "ラット", "R2": "ラット"}
    f = lp.is_factors(_raw(), IS, used={"M1", "M2", "R1"}, groups=groups)
    assert f["f0"] == pytest.approx(110 / 100)  # 空欄の試料どうし
    assert f["f3"] == pytest.approx(300 / 340)  # 使わない R2 は平均に入らないが、係数は付く


def test_is_groups_from_meta():
    meta = pd.DataFrame({lp.IS_GROUP_COL: ["マウス", " ", "ラット", np.nan]}, index=["a", "b", "c", "d"])
    assert lp.is_groups(meta) == {"a": "マウス", "c": "ラット"}
