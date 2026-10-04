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
    df["conc_text"] = df.get("conc", "").astype(str)  # 入力エラー (数値として読めない値) の判定用に元の文字を残す
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


# ---------------------------------------------------------------- 検量線の評価と定量下限の確認
def calibration_detail(full, max_bias=20.0, max_cv=20.0):
    """化合物ごとの検量線の情報と、正確さ・精度を実測で確認した定量下限 (LLOQ)。

    - 使用した点: 検量点 (*) が付き、濃度が算出された STD
    - 除外した点: 検量点が付いていない、または濃度が算出されなかった STD
    - 検量線の式・重み付け: LabSolutions のエクスポートに含まれないため「未取得」
    - 確認済み LLOQ: 繰り返し測定 (n >= 2) があり、逆算濃度の平均の偏りが ±max_bias% 以内かつ CV が max_cv% 以下の
      最も低い濃度。繰り返しが無い濃度は精度を評価できないため確認済みにしない
    - 推定 LOQ: LabSolutions が S/N = 10 から算出した値 (試料の中央値)。実測確認ではない
    """
    std = full[full["is_std"] & full["set_conc"].notna()]
    rows = []
    for c, g in std.groupby("compound"):
        used = g[g["cal_point"] & g["conc"].notna()]
        excluded = g[~(g["cal_point"] & g["conc"].notna())]
        levels = []
        for lv, h in used.groupby("set_conc"):
            bias = 100 * (h["conc"].mean() / lv - 1)
            cv = 100 * h["conc"].std() / h["conc"].mean() if len(h) >= 2 else np.nan
            levels.append((lv, len(h), bias, cv))
        verified, basis = None, "繰り返し測定のある濃度が無いため未確認"
        for lv, n, bias, cv in sorted(levels):
            if n < 2:
                continue
            if abs(bias) <= max_bias and cv <= max_cv:
                verified, basis = lv, f"{lv:g} (n={n}, 偏り {bias:+.1f}%, CV {cv:.1f}%)"
                break
            basis = f"{lv:g} で基準外 (n={n}, 偏り {bias:+.1f}%, CV {cv:.1f}%)"
        samples = full[(~full["is_std"]) & (full["compound"] == c) & (full["loq"] > 0)]
        rows.append({
            "化合物": c,
            "使用した点": ", ".join(f"{lv:g} (n={n})" for lv, n, _, _ in sorted(levels)) or "なし",
            "除外した点": ", ".join(f"{r['file']}" for _, r in excluded.iterrows()) or "なし",
            "濃度レベル数": len(levels),
            "検量線の式・重み付け": "未取得 (エクスポートに含まれない)",
            "逆算誤差 (各濃度, %)": ", ".join(f"{lv:g}: {b:+.1f}" for lv, _, b, _ in sorted(levels)),
            "最低濃度の CV%": next((cv for lv, n, b, cv in sorted(levels)), np.nan),
            "確認済み LLOQ": f"{verified:g}" if verified is not None else "未確認",  # 表示のため文字列で統一
            "確認の根拠": basis,
            "推定 LOQ (S/N=10, 中央値)": samples["loq"].median() if len(samples) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- QC 試料の評価
QC_KNOWN, QC_POOL = "既知濃度 QC", "プール QC"
BLANKS = ["溶媒ブランク", "処理ブランク", "キャリーオーバー確認ブランク"]
NOT_EVALUATED = "未評価"


def qc_evaluation(full, sample_type, inj, thresholds=None, nominal=None, lloq=None):
    """QC 試料の評価。sample_type: {label: 試料種別}、thresholds: 化合物 -> (CV 上限%, 偏り上限%, キャリーオーバー上限%)。

    - 既知濃度 QC: 偏り (平均 / 理論濃度 - 1) と精度 (CV)。理論濃度 (nominal) が無い化合物は 未評価
    - プール QC: 精度 (CV) と測定順のドリフト (Spearman)。真値が不明なので正確さは評価しない
    - ブランク: 検出された値。キャリーオーバー確認ブランクは LLOQ に対する割合 (FDA / ICH M10 の目安 20%)
    QC が無い項目は「未評価」とし、合格扱いにしない。
    """
    from scipy import stats as _st

    thresholds = thresholds or {}
    d = full[~full["is_std"]].copy()
    d["種別"] = d[LABEL_COL].map(sample_type).fillna("試料")
    d = d.merge(inj[["file", "order"]], on="file", how="left")
    compounds = sorted(full["compound"].unique())
    rows = []
    for c in compounds:
        cv_max, bias_max, carry_max = thresholds.get(c, (15.0, 15.0, 20.0))
        row = {"化合物": c}
        pool = d[(d["種別"] == QC_POOL) & (d["compound"] == c)]
        vals = pool["conc"].dropna()
        if len(vals) >= 3:
            cv = 100 * vals.std() / vals.mean()
            rho = _st.spearmanr(pool.dropna(subset=["conc"])["order"], vals)[0] if len(vals) >= 4 else np.nan
            row.update({"プール QC n": len(vals), "プール QC CV%": cv, "プール QC ドリフト ρ": rho,
                        "プール QC 判定": "合格" if cv <= cv_max else "不合格"})
        else:
            row.update({"プール QC n": len(vals), "プール QC 判定": NOT_EVALUATED if len(pool) == 0 else "未評価 (n<3)"})
        known = d[(d["種別"] == QC_KNOWN) & (d["compound"] == c)]
        nom = (nominal or {}).get(c)
        kv = known["conc"].dropna()
        if len(kv) and nom:
            bias = 100 * (kv.mean() / nom - 1)
            cv = 100 * kv.std() / kv.mean() if len(kv) >= 2 else np.nan
            ok = abs(bias) <= bias_max and (np.isnan(cv) or cv <= cv_max)
            row.update({"既知濃度 QC n": len(kv), "既知濃度 QC 偏り%": bias, "既知濃度 QC CV%": cv,
                        "既知濃度 QC 判定": ("合格" if ok else "不合格") + (" (精度は未評価)" if np.isnan(cv) else "")})
        else:
            row.update({"既知濃度 QC n": len(kv),
                        "既知濃度 QC 判定": NOT_EVALUATED + (" (理論濃度が未入力)" if len(kv) else "")})
        blanks = d[d["種別"].isin(BLANKS) & (d["compound"] == c)]
        if len(blanks):
            det = blanks["conc"].dropna()
            row["ブランク 検出数"] = f"{len(det)} / {len(blanks)}"
            row["ブランク 最大値"] = det.max() if len(det) else np.nan
            carry = blanks[blanks["種別"] == "キャリーオーバー確認ブランク"]["conc"].dropna()
            ll = (lloq or {}).get(c)
            if len(blanks[blanks["種別"] == "キャリーオーバー確認ブランク"]) and isinstance(ll, (int, float)):
                pct = 100 * (carry.max() if len(carry) else 0) / ll
                row["キャリーオーバー (% of LLOQ)"] = pct
                row["キャリーオーバー判定"] = "合格" if pct <= carry_max else "不合格"
            else:
                row["キャリーオーバー判定"] = NOT_EVALUATED
        else:
            row.update({"ブランク 検出数": NOT_EVALUATED, "キャリーオーバー判定": NOT_EVALUATED})
        rows.append(row)
    return pd.DataFrame(rows)


def known_qc_table(full, sample_type, nominal, max_bias=15.0, max_cv=15.0):
    """既知濃度 QC の偏りと精度 (QC の label ごと)。nominal: {QC の label: {化合物: 理論濃度}}"""
    d = full[~full["is_std"]]
    rows = []
    for lb in [k for k, v in sample_type.items() if v == QC_KNOWN]:
        for c, nom in (nominal.get(lb) or {}).items():
            v = d[(d[LABEL_COL] == lb) & (d["compound"] == c)]["conc"].dropna()
            if not nom or not len(v):
                rows.append({"QC": lb, "化合物": c, "n": len(v), "理論濃度": nom, "判定": NOT_EVALUATED})
                continue
            bias = 100 * (v.mean() / nom - 1)
            cv = 100 * v.std() / v.mean() if len(v) >= 2 else np.nan
            ok = abs(bias) <= max_bias and (np.isnan(cv) or cv <= max_cv)
            rows.append({"QC": lb, "化合物": c, "n": len(v), "理論濃度": nom, "平均": v.mean(), "偏り%": bias,
                         "CV%": cv, "判定": ("合格" if ok else "不合格") + (" (精度は未評価: n<2)" if np.isnan(cv) else "")})
    return pd.DataFrame(rows)
