"""デモ用の脂質データ (模擬データ) を作る。

LC-MS/MS の定量結果と同じサンプル (label) について、肝臓 TG・血中 TG・総コレステロール・LDL-C・HDL-C・FFA を
乱数で生成する。デモで相関や正準相関分析に関係が現れるよう、一部の項目は実測の代謝物と連動させている。
値はすべて架空であり、実験結果ではない。

    uv run scripts/make_demo_lipid.py data/20260925_Fujita.txt              # -> data/demo_lipid_simulated.csv
    uv run scripts/make_demo_lipid.py data/20260925_Fujita.txt -o out.csv --seed 1

生成のしかた (z はサンプル間で標準化した値、e は独立な標準正規乱数):
    BCAA 因子 = z(log(Val + Leu + Ile))、ケトン因子 = z(log 3-ヒドロキシ酪酸)   (IS 補正後・希釈なし)
    肝臓 TG   = 0.65 BCAA + 0.76 e
    血中 TG   = 0.55 肝臓 TG + 0.84 e
    LDL-C     = 0.45 BCAA + 0.89 e
    HDL-C     = -0.50 BCAA + 0.87 e
    FFA       = 0.60 ケトン + 0.80 e
    各 z を対数正規分布で下の平均・変動係数の値に変換し、
    T-Cho     = HDL-C + LDL-C + 血中 TG / 5 + 誤差   (Friedewald の式と整合させる)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import lcms_analysis as la  # noqa: E402
from make_conc_table import DILUTION_COL, IS_NAME, LABEL_COL, is_std  # noqa: E402

# マウス (通常食) で一般的な範囲を想定した平均と変動係数
SCALES = {
    "肝臓TG (mg/g 組織)": (25.0, 0.35),
    "血中TG (mg/dL)": (90.0, 0.30),
    "LDL-C (mg/dL)": (15.0, 0.35),
    "HDL-C (mg/dL)": (60.0, 0.18),
    "FFA (mEq/L)": (0.90, 0.30),
}


def zscore(v):
    v = np.asarray(v, dtype=float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def lognormal(z, mean, cv):
    s = np.sqrt(np.log(1 + cv ** 2))
    return mean * np.exp(s * z - s ** 2 / 2)


def make(lcms_path, seed=20260925):
    raw, is_df, _ = la.load_tables(Path(lcms_path).read_bytes(), IS_NAME)
    df = is_df[(is_df[DILUTION_COL] == "") & ~is_df[LABEL_COL].map(is_std)].drop_duplicates(LABEL_COL)
    df = df.set_index(LABEL_COL).loc[sorted(df[LABEL_COL], key=la.natural_key)]
    rng = np.random.default_rng(seed)
    n = len(df)
    bcaa = zscore(np.log(df[["Valine", "Leucine", "Isoleucine"]].sum(axis=1, min_count=3)))
    ketone = np.nan_to_num(zscore(np.log(df["3-ヒドロキシ酪酸"])))
    bcaa = np.nan_to_num(bcaa)
    e = lambda: rng.standard_normal(n)  # noqa: E731

    liver = 0.65 * bcaa + 0.76 * e()
    z = {
        "肝臓TG (mg/g 組織)": liver,
        "血中TG (mg/dL)": 0.55 * liver + 0.84 * e(),
        "LDL-C (mg/dL)": 0.45 * bcaa + 0.89 * e(),
        "HDL-C (mg/dL)": -0.50 * bcaa + 0.87 * e(),
        "FFA (mEq/L)": 0.60 * ketone + 0.80 * e(),
    }
    out = pd.DataFrame({"SampleID": df.index})
    for col, (mean, cv) in SCALES.items():
        out[col] = lognormal(z[col], mean, cv)
    out["T-Cho (mg/dL)"] = out["HDL-C (mg/dL)"] + out["LDL-C (mg/dL)"] + out["血中TG (mg/dL)"] / 5 + rng.normal(0, 4, n)
    cols = ["SampleID", "肝臓TG (mg/g 組織)", "血中TG (mg/dL)", "T-Cho (mg/dL)", "LDL-C (mg/dL)", "HDL-C (mg/dL)",
            "FFA (mEq/L)"]
    out = out[cols].round({"肝臓TG (mg/g 組織)": 1, "血中TG (mg/dL)": 0, "T-Cho (mg/dL)": 0, "LDL-C (mg/dL)": 1,
                           "HDL-C (mg/dL)": 0, "FFA (mEq/L)": 2})
    # 欠損の扱いも確認できるよう、1 サンプルの FFA を欠損 (溶血で測定不可を想定) にする
    out.loc[out["SampleID"] == "D-7", "FFA (mEq/L)"] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lcms", help="LabSolutions の定量結果 (.txt)")
    ap.add_argument("-o", "--output", default=None, help="出力 CSV (既定: data/demo_lipid_simulated.csv)")
    ap.add_argument("--seed", type=int, default=20260925, help="乱数の種 (既定: 20260925)")
    args = ap.parse_args()
    out = make(args.lcms, args.seed)
    path = Path(args.output) if args.output else Path(args.lcms).parent / "demo_lipid_simulated.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{len(out)} サンプル x {out.shape[1] - 1} 項目 -> {path}  (模擬データ)")


if __name__ == "__main__":
    main()
