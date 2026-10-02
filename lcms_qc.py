"""QC 用の解析: 生データの全列の読み込み、検量線範囲による希釈の選択、STD の正確さ、測定順ドリフト。"""
import numpy as np
import pandas as pd
from scipy import stats

from make_conc_table import DILUTION_COL, FILE_COL, LABEL_COL, MISSING, decode_bytes, parse_fname

# LabSolutions の列名 -> 内部名
FULL_COLS = {
    "データファイル名": "file", "サンプルタイプ": "sample_type", "保持時間": "rt", "面積": "area", "濃度": "conc",
    "S/N": "sn", "定量限界": "loq", "検出限界": "lod", "設定濃度": "set_conc", "正確さ%": "accuracy",
    "検量点": "cal_point", "誤差%": "error", "分析日時": "datetime",
}
NUMERIC = ["rt", "area", "conc", "sn", "loq", "lod", "set_conc", "accuracy", "error"]
STD_TYPE = "標準(検量点)"

SRC_NONE, SRC_DIL = "希釈なし", "x10"
CHOICE_AUTO = "自動 (化合物単位)"         # 1 サンプルでも上限を超えたら、その化合物は全サンプル希釈測定を使う
CHOICE_AUTO_SAMPLE = "自動 (サンプル単位)"  # サンプルごとに上限を超えたものだけ希釈測定を使う


def parse_full(raw_bytes):
    """全化合物・全列を long 形式 (1 行 = 化合物 x データファイル) で返す。"""
    rows, name, header = [], None, None
    for line in decode_bytes(raw_bytes).splitlines():
        cells = line.split("\t")
        if cells[0] == "ID#":
            name, header = None, None
        elif cells[0] == "Name":
            name = cells[1].strip()
        elif "データファイル名" in cells:
            header = cells
        elif name is not None and header is not None and cells[0].strip().isdigit():
            row = {FULL_COLS[h]: v.strip() for h, v in zip(header, cells) if h in FULL_COLS}
            row["compound"] = name
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c].replace({MISSING: np.nan, "(INF)": np.inf}), errors="coerce")
    df["cal_point"] = df.get("cal_point", "").eq("*")
    df["datetime"] = pd.to_datetime(df.get("datetime"), format="%Y/%m/%d %H:%M:%S", errors="coerce")
    meta = df["file"].map(parse_fname)
    df[LABEL_COL] = meta.str[0]
    df[DILUTION_COL] = meta.str[1]
    df["is_std"] = df["sample_type"].eq(STD_TYPE)
    return df


def injection_order(full):
    """データファイルごとの分析日時と測定順 (1 始まり)。"""
    inj = full.groupby("file", as_index=False).agg(
        datetime=("datetime", "first"), sample_type=("sample_type", "first"), is_std=("is_std", "first"),
        label=(LABEL_COL, "first"), dilution=(DILUTION_COL, "first"))
    inj = inj.sort_values("datetime").reset_index(drop=True)
    inj["order"] = np.arange(1, len(inj) + 1)
    return inj.rename(columns={"label": LABEL_COL, "dilution": DILUTION_COL})


# ---------------------------------------------------------------- 検量線範囲
def calibration_ranges(full):
    """化合物ごとの検量線範囲 (検量点に使われ、濃度が算出された STD の設定濃度の最小〜最大)。"""
    std = full[full["is_std"] & full["cal_point"] & full["conc"].notna() & full["set_conc"].notna()]
    rng = std.groupby("compound")["set_conc"].agg(下限="min", 上限="max", 検量点数="count")
    levels = std.groupby("compound")["set_conc"].apply(lambda s: ", ".join(f"{v:g}" for v in sorted(s.unique())))
    return rng.assign(濃度レベル=levels, 一点検量=std.groupby("compound")["set_conc"].nunique() <= 1)


def _pick(row_u, row_d, lo, hi, choice):
    """1 サンプル・1 化合物について (採用元, フラグ) を返す。row_* は (行が存在するか, 生の濃度)。"""
    has_u, u = row_u
    has_d, d = row_d
    if choice == SRC_NONE:
        src = SRC_NONE if has_u else (SRC_DIL if has_d else None)
    elif choice == SRC_DIL:
        src = SRC_DIL if has_d else (SRC_NONE if has_u else None)
    else:  # 自動 (サンプル単位): 希釈なしが上限を超えたら希釈測定を採用
        if has_u and pd.notna(u) and (np.isnan(hi) or u <= hi):
            src = SRC_NONE
        elif has_d and pd.notna(d):
            src = SRC_DIL
        elif has_u:
            src = SRC_NONE
        else:
            src = SRC_DIL if has_d else None
    if src is None:
        return None, "測定なし"
    v = u if src == SRC_NONE else d
    if pd.isna(v):
        flag = "未検出"
    elif not np.isnan(hi) and v > hi:
        flag = "上限超過"
    elif not np.isnan(lo) and v < lo:
        flag = "下限未満"
    else:
        flag = "範囲内"
    return src, flag


def merge_dilutions(values, raw, compounds, ranges, choices, factor=10.0):
    """希釈なし / 希釈測定を化合物ごとに選び、1 label 1 行の表にまとめる。

    values: 解析に使う値の表 (IS 補正後 or 補正前)。STD は除いておく
    raw: 補正前の表 (検量線範囲との比較は装置で測った生の濃度で行う)
    choices: {化合物: "自動" / "希釈なし" / "x10"}
    戻り値: (統合後の表, 採用元の表, フラグの表)
    """
    def by_label(df, dil):
        sub = df[df[DILUTION_COL] == dil]
        return sub.drop_duplicates(LABEL_COL).set_index(LABEL_COL)

    dil_name = next((d for d in values[DILUTION_COL].unique() if d), SRC_DIL)
    v_u, v_d = by_label(values, ""), by_label(values, dil_name)
    r_u, r_d = by_label(raw, ""), by_label(raw, dil_name)
    labels = list(dict.fromkeys(values[LABEL_COL]))
    out = pd.DataFrame(index=labels, columns=compounds, dtype=float)
    src = pd.DataFrame("", index=labels, columns=compounds)
    flag = pd.DataFrame("", index=labels, columns=compounds)
    for c in compounds:
        lo, hi = (ranges.loc[c, "下限"], ranges.loc[c, "上限"]) if c in ranges.index else (np.nan, np.nan)
        choice = choices.get(c, CHOICE_AUTO)
        if choice == CHOICE_AUTO:
            # 希釈測定があるサンプルのうち 1 つでも上限を超えたら、化合物全体で希釈測定に揃える
            # (希釈なし / 希釈測定の間に系統差があると、サンプルごとに混ぜると見かけの差が生じるため)
            over = [lb for lb in labels if lb in r_u.index and lb in r_d.index and pd.notna(hi)
                    and pd.notna(r_u.at[lb, c]) and r_u.at[lb, c] > hi]
            choice = SRC_DIL if over else SRC_NONE
        for lb in labels:
            s, f = _pick((lb in r_u.index, r_u[c].get(lb, np.nan) if c in r_u else np.nan),
                         (lb in r_d.index, r_d[c].get(lb, np.nan) if c in r_d else np.nan),
                         lo, hi, choice)
            src.loc[lb, c], flag.loc[lb, c] = s or "", f
            if s == SRC_NONE and lb in v_u.index:
                out.loc[lb, c] = v_u.loc[lb, c]
            elif s == SRC_DIL and lb in v_d.index:
                out.loc[lb, c] = v_d.loc[lb, c] * factor
    files = {lb: " / ".join(values.loc[values[LABEL_COL] == lb, FILE_COL]) for lb in labels}
    merged = out.reset_index(names=LABEL_COL)
    merged.insert(0, FILE_COL, merged[LABEL_COL].map(files))
    merged.insert(2, DILUTION_COL, "統合")
    return merged, src.rename_axis(LABEL_COL), flag.rename_axis(LABEL_COL)


def dilution_summary(raw, compounds, ranges):
    """化合物ごとに、希釈なし測定が検量線上限を超えたサンプル数などをまとめる。"""
    u = raw[(raw[DILUTION_COL] == "")]
    rows = []
    for c in compounds:
        lo, hi = (ranges.loc[c, "下限"], ranges.loc[c, "上限"]) if c in ranges.index else (np.nan, np.nan)
        v = u[c].dropna()
        rows.append({"化合物": c, "下限": lo, "上限": hi, "検出数 (希釈なし)": len(v),
                     "上限超過": int((v > hi).sum()) if pd.notna(hi) else 0,
                     "下限未満": int((v < lo).sum()) if pd.notna(lo) else 0,
                     "最大値 (希釈なし)": v.max() if len(v) else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- STD の正確さ
def std_accuracy(full):
    """STD の正確さ (%) の long 表。FDA (2018) / ICH M10 (2022) の基準 (±15%, 最低濃度は ±20%) で判定する。"""
    std = full[full["is_std"]].copy()
    lowest = std[std["set_conc"].notna()].groupby("compound")["set_conc"].transform("min")
    std["最低濃度"] = std["set_conc"].eq(lowest.reindex(std.index))
    std["許容幅%"] = np.where(std["最低濃度"], 20, 15)
    n_levels = std[std["cal_point"]].groupby("compound")["set_conc"].nunique()
    one_point = std["compound"].map(n_levels).fillna(0) <= 1
    std["判定"] = np.select(
        [std["accuracy"].isna(), one_point, (std["accuracy"] - 100).abs() <= std["許容幅%"]],
        ["未算出", "評価不可 (1点検量)", "合格"], "不合格")
    return std


def accuracy_summary(acc):
    """化合物ごとの合否の集計。"""
    calc = acc[acc["判定"].isin(["合格", "不合格"])]
    s = calc.groupby("compound").agg(
        STD数=("判定", "size"), 合格=("判定", lambda x: int((x == "合格").sum())),
        正確さ_平均=("accuracy", "mean"), 正確さ_最小=("accuracy", "min"), 正確さ_最大=("accuracy", "max"))
    s["合格率%"] = (100 * s["合格"] / s["STD数"]).round(1)
    # FDA / ICH M10: 検量線の標準のうち 75% 以上 (かつ 6 点以上) が基準内であること
    s["検量線基準 (≥75%)"] = np.where(s["合格率%"] >= 75, "満たす", "満たさない")
    one = acc.loc[acc["判定"] == "評価不可 (1点検量)", "compound"].unique()
    s = s.reindex(s.index.union(one))
    s.loc[one, "検量線基準 (≥75%)"] = "評価不可 (1点検量)"
    return s.reset_index(names="化合物")


# ---------------------------------------------------------------- ドリフト
def drift_table(full, inj, value="area", groups=("", "x10")):
    """化合物ごとに、測定順と値の Spearman 相関 (サンプルのみ, 希釈グループ別) と CV% を求める。"""
    df = full[~full["is_std"]].merge(inj[["file", "order"]], on="file")
    rows = []
    for (c, dil), g in df.groupby(["compound", DILUTION_COL]):
        g = g[np.isfinite(g[value])]
        if len(g) < 4:
            continue
        rho, p = stats.spearmanr(g["order"], g[value])
        slope = np.polyfit(g["order"], g[value], 1)[0]
        mean = g[value].mean()
        rows.append({"化合物": c, "希釈": dil or SRC_NONE, "n": len(g), "Spearman ρ": rho, "p": p,
                     "傾き (%/測定)": 100 * slope / mean if mean else np.nan,
                     "CV%": 100 * g[value].std() / mean if mean else np.nan})
    return pd.DataFrame(rows)


def dilution_linearity(raw, compounds, factor=10.0, min_n=3):
    """希釈の直線性: 両方で検出されたサンプルについて (希釈測定 × 希釈倍率) / 希釈なし の比を化合物ごとにまとめる。

    装置の算出濃度 (IS 補正前) で比べる。比が 1 から大きく外れる化合物は、希釈による応答の非直線性・
    マトリックス効果・測定までの時間による変化 (希釈測定を先に測っている場合) などを疑う。
    """
    u = raw[raw[DILUTION_COL] == ""].drop_duplicates(LABEL_COL).set_index(LABEL_COL)
    dil = next((d for d in raw[DILUTION_COL].unique() if d), None)
    if dil is None:
        return pd.DataFrame(), pd.DataFrame()
    d = raw[raw[DILUTION_COL] == dil].drop_duplicates(LABEL_COL).set_index(LABEL_COL)
    labels = u.index.intersection(d.index)
    rows, pairs = [], []
    for c in compounds:
        a, b = u.loc[labels, c], d.loc[labels, c] * factor
        ok = a.notna() & b.notna() & (a > 0) & (b > 0)
        if ok.sum() == 0:
            continue
        ratio = b[ok] / a[ok]
        pairs.append(pd.DataFrame({"化合物": c, LABEL_COL: ratio.index, "希釈なし": a[ok].values,
                                   "希釈測定 x 倍率": b[ok].values, "比": ratio.values}))
        if ok.sum() >= min_n:
            rows.append({"化合物": c, "n": int(ok.sum()), "比の中央値": ratio.median(),
                         "第1四分位": ratio.quantile(0.25), "第3四分位": ratio.quantile(0.75),
                         "比の CV%": 100 * ratio.std() / ratio.mean()})
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        tbl["判定"] = np.where((tbl["比の中央値"] - 1).abs() <= 0.15, "±15% 以内", "要確認")
    return tbl, (pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame())


# ---------------------------------------------------------------- 定量限界 (LOQ) / 検出限界 (LOD)
# LabSolutions は測定ごとに S/N から LOD (S/N = 3) と LOQ (S/N = 10) を濃度で出力する。
# ノイズが 0 (S/N が ∞) のときは LOD = LOQ = 0 と出力されるので「評価不可」とする。
LIM_ND, LIM_UNK, LIM_LOD, LIM_LOQ, LIM_OK = "未検出", "評価不可 (S/N = ∞)", "LOD 未満", "LOD 以上 LOQ 未満", "LOQ 以上"
LIM_ORDER = [LIM_ND, LIM_LOD, LIM_LOQ, LIM_OK, LIM_UNK]
MASK_NONE, MASK_LOD, MASK_LOQ = "マスクしない", "検出限界 (LOD) 未満", "定量限界 (LOQ) 未満"
REPL_NAN, REPL_HALF, REPL_LIM = "欠損 (NaN) にする", "限界値の 1/2 に置き換え", "限界値に置き換え"


def limit_categories(full):
    """各測定 (化合物 x ファイル) を 未検出 / LOD 未満 / LOD〜LOQ / LOQ 以上 / 評価不可 に分類した列を足す。"""
    df = full.copy()
    unknown = ~(df["lod"] > 0) | ~(df["loq"] > 0) | np.isinf(df["sn"].fillna(0))
    df["区分"] = np.select(
        [df["conc"].isna(), unknown, df["conc"] < df["lod"], df["conc"] < df["loq"]],
        [LIM_ND, LIM_UNK, LIM_LOD, LIM_LOQ], LIM_OK)
    return df


def limit_summary(cat, samples_only=True):
    df = cat[~cat["is_std"]] if samples_only else cat
    tbl = df.pivot_table(index="compound", columns="区分", values="file", aggfunc="count", fill_value=0)
    tbl = tbl.reindex(columns=LIM_ORDER, fill_value=0)
    med = df[df["loq"] > 0].groupby("compound")[["lod", "loq"]].median().rename(
        columns={"lod": "LOD (中央値)", "loq": "LOQ (中央値)"})
    out = tbl.join(med)
    detected = out[[LIM_LOD, LIM_LOQ, LIM_OK, LIM_UNK]].sum(axis=1)
    out["LOQ 未満の割合% (検出のうち)"] = (100 * (out[LIM_LOD] + out[LIM_LOQ]) / detected.replace(0, np.nan)).round(1)
    return out.reset_index(names="化合物")


def apply_limit_mask(df, full, compounds, level, replace, skip=()):
    """LOD / LOQ 未満の値をマスクする。判定は装置の算出濃度 (IS 補正前) と測定ごとの限界値で行う。

    df が IS 補正後の表 (比率の列を持つ) なら、置き換える値も同じ比率で補正する。
    戻り値: (マスク後の表, マスクしたセル数)
    """
    if level == MASK_NONE:
        return df, 0
    col = "lod" if level == MASK_LOD else "loq"
    lim = full.pivot_table(index="file", columns="compound", values=col, aggfunc="first")
    raw = full.pivot_table(index="file", columns="compound", values="conc", aggfunc="first")
    out = df.copy()
    ratio = out["IS比率"] if "IS比率" in out else pd.Series(1.0, index=out.index)
    n = 0
    for c in compounds:
        if c in skip or c not in lim:
            continue
        L = out[FILE_COL].map(lim[c])
        R = out[FILE_COL].map(raw[c])
        m = R.notna() & (L > 0) & (R < L)
        if not m.any():
            continue
        n += int(m.sum())
        if replace == REPL_NAN:
            out.loc[m, c] = np.nan
        else:
            f = 0.5 if replace == REPL_HALF else 1.0
            out.loc[m, c] = (L * f / ratio)[m]
    return out, n
