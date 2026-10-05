"""GUI (app.py) で使う解析ロジック。Streamlit には依存しない。"""
import re

import numpy as np
import pandas as pd

from make_conc_table import (
    DILUTION_COL,
    FILE_COL,
    LABEL_COL,
    MISSING,
    RATIO_COL,
    decode_bytes,
    is_std,
    normalize_by_is,
    parse_blocks,
    parse_fname,
)

GROUP_COL = "希釈"          # 表示用: "" -> "希釈なし"
NO_DILUTION = "希釈なし"
COND_COL = "condition"
DATASET_COL = "データセット"
CPD_COL = "化合物"
VALUE_COL = "値"

# 値の種類
CONC = "濃度"
FC = "Fold Change"
LOG2FC = "log2(Fold Change)"
LOG2CONC = "log2(濃度)"
LOG_NONE, LOG2, LOG10 = "なし", "log2", "log10"
LOG_CHOICES = [LOG2, LOG10, LOG_NONE]
# 要約統計量
MEAN_SD = "平均 ± SD"
MEDIAN_IQR = "中央値 + IQR"


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def group_name(dilution):
    return dilution or NO_DILUTION


def group_order(groups):
    # 希釈なし -> x10 -> x100 ... の順
    return sorted(set(groups), key=lambda g: (g != NO_DILUTION, natural_key(g)))


def unique_names(labels):
    """重複ラベル (STD の繰り返し測定など) に #2, #3 を付けて一意にする。"""
    seen, out = {}, []
    for lb in labels:
        seen[lb] = seen.get(lb, 0) + 1
        out.append(lb if seen[lb] == 1 else f"{lb} #{seen[lb]}")
    return out


# ---------------------------------------------------------------- 読み込み
def load_tables(raw_bytes, is_name):
    """(補正前の表, IS 補正後の表 or None, 化合物リスト) を返す。"""
    blocks, files = parse_blocks(decode_bytes(raw_bytes))
    compounds = list(blocks)
    meta = {f: parse_fname(f) for f in files}

    def num(v):
        # 数値として読めない値 (入力エラー) は欠損にする。状態は lcms_pipeline で「入力エラー」として記録する
        try:
            return np.nan if v in ("", MISSING) else float(v)
        except ValueError:
            return np.nan

    raw = pd.DataFrame(
        [[f, *meta[f]] + [num(blocks[c].get(f, "")) for c in compounds] for f in files],
        columns=[FILE_COL, LABEL_COL, DILUTION_COL] + compounds,
    )

    corrected = None
    if is_name in blocks:
        rows = []
        for f, ratio, conc in normalize_by_is(files, blocks, meta, is_name):
            # 内部標準が未検出のサンプルは比率が None になる (補正できないので欠損として扱う)
            rows.append([f, *meta[f], np.nan if ratio is None else ratio] + [conc.get(c, np.nan) for c in compounds])
        corrected = pd.DataFrame(rows, columns=[FILE_COL, LABEL_COL, DILUTION_COL, RATIO_COL] + compounds)
        corrected[compounds] = corrected[compounds].astype(float).round(3)
        corrected[RATIO_COL] = corrected[RATIO_COL].astype(float).round(4)
    return raw, corrected, compounds


def with_condition(df, cond_map):
    """label の右隣に condition 列を挿入したコピーを返す。"""
    out = df.copy()
    out.insert(out.columns.get_loc(LABEL_COL) + 1, COND_COL, out[LABEL_COL].map(cond_map).fillna(""))
    return out


def build_base(df, compounds, cond_map, include_std, mult_dilution):
    """可視化・解析用の表 (希釈グループ列・condition 列付き) を作る。"""
    base = df.copy()
    if not include_std:
        base = base[~base[LABEL_COL].map(is_std)]
    base[GROUP_COL] = base[DILUTION_COL].map(group_name)
    base[COND_COL] = base[LABEL_COL].map(cond_map).fillna("").astype(str).str.strip()
    if mult_dilution:
        factor = base[DILUTION_COL].str.extract(r"x(\d+)")[0].astype(float).fillna(1)
        base[compounds] = base[compounds].mul(factor, axis=0)
    return base


def condition_order(cond_map, labels, control=None):
    """condition を (対照群を先頭に) 入力順で並べる。"""
    conds = []
    for lb in sorted(labels, key=natural_key):
        c = str(cond_map.get(lb, "") or "").strip()
        if c and c not in conds:
            conds.append(c)
    if control in conds:
        conds.remove(control)
        conds.insert(0, control)
    return conds


def order_samples(df, conds):
    """condition 順 -> label の自然順 でサンプルを並べる (condition 未設定は末尾)。"""
    rank = {c: i for i, c in enumerate(conds)}
    keys = [(rank.get(c, len(rank)), natural_key(lb)) for c, lb in zip(df[COND_COL], df[LABEL_COL])]
    return df.iloc[sorted(range(len(df)), key=keys.__getitem__)]


# ---------------------------------------------------------------- Fold Change
def control_means(base, compounds, control):
    """希釈グループごとの対照群平均 (行 = 希釈グループ, 列 = 化合物)。"""
    ctrl = base[base[COND_COL] == control]
    return ctrl.groupby(GROUP_COL)[compounds].mean()


def to_fold_change(base, compounds, control):
    """各サンプルの値を、同じ希釈グループの対照群平均で割った表を返す。
    対照群で未検出 (平均が NaN または 0) の化合物は NaN。"""
    means = control_means(base, compounds, control).replace(0, np.nan)
    out = base.copy()
    for g, idx in out.groupby(GROUP_COL).groups.items():
        m = means.loc[g] if g in means.index else pd.Series(np.nan, index=compounds)
        out.loc[idx, compounds] = out.loc[idx, compounds].div(m, axis=1)
    return out


def log2_safe(x):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log2(np.where(x > 0, x, 1)), np.nan)


def transform_values(base, compounds, kind, control):
    """kind (濃度 / log2(濃度) / FC / log2FC) に応じてサンプルごとの値を変換する。"""
    if kind == CONC:
        return base
    if kind == LOG2CONC:
        out = base.copy()
        out[compounds] = log2_safe(out[compounds].values)
        return out
    out = to_fold_change(base, compounds, control)
    if kind == LOG2FC:
        out[compounds] = log2_safe(out[compounds].values)
    return out


# ---------------------------------------------------------------- 要約統計
def summarize(df, compounds, stat, by):
    """by ごと・化合物ごとの要約 (中心値と誤差範囲の上下端) を long 形式で返す。"""
    long = df.melt(id_vars=by, value_vars=compounds, var_name=CPD_COL, value_name=VALUE_COL)
    g = long.groupby(by + [CPD_COL], sort=False)[VALUE_COL]
    s = g.agg(n="count", mean="mean", sd="std", median="median",
              q1=lambda v: v.quantile(0.25), q3=lambda v: v.quantile(0.75)).reset_index()
    if stat == MEAN_SD:
        s["center"], s["lower"], s["upper"] = s["mean"], s["mean"] - s["sd"], s["mean"] + s["sd"]
    else:
        s["center"], s["lower"], s["upper"] = s["median"], s["q1"], s["q3"]
    return s


def summary_values(df, compounds, kind, control, stat, by):
    """条件ごとの要約とサンプルごとの点を返す。

    Fold Change はサンプル値 / 対照群平均 で計算し、その平均 (= 条件平均 / 対照群平均) や
    中央値を中心値とする。log2(FC) は FC スケールで要約した中心値・誤差範囲を log2 変換する
    (平均 ± SD の下端が 0 以下になる場合、その側の誤差棒は描かない)。
    """
    lin_kind = FC if kind == LOG2FC else kind
    vals = transform_values(df, compounds, lin_kind, control)
    summ = summarize(vals, compounds, stat, by)
    points = vals.melt(id_vars=by + [LABEL_COL, FILE_COL], value_vars=compounds, var_name=CPD_COL, value_name=VALUE_COL)
    if kind == LOG2FC:
        for c in ["center", "lower", "upper"]:
            summ[c] = log2_safe(summ[c])
        points[VALUE_COL] = log2_safe(points[VALUE_COL])
    return summ, points.dropna(subset=[VALUE_COL])


# ---------------------------------------------------------------- 行列の前処理
def zscore_cols(X):
    sd = X.std(ddof=0)
    keep = sd > 0
    return (X.loc[:, keep] - X.loc[:, keep].mean()) / sd[keep]


def log_safe(X, base=10):
    """対数変換 (base = 2 または 10)。0 は最小の正の値の 1/2 に置き換える。負の値を含む列 (Z スコアなど) は変換しない。"""
    X = X.astype(float)
    cols = [c for c in X.columns if not (X[c] < 0).any()]
    if not cols:
        return X
    P = X[cols]
    pos = P[P > 0].min().min()
    out = X.copy()
    if pd.notna(pos):
        out[cols] = np.log(P.clip(lower=pos / 2)) / np.log(base)
    return out


def log10_safe(X):
    return log_safe(X, 10)


def apply_log(X, mode):
    """mode: "log2" / "log10" / "なし" (True は log10 とみなす: 以前の版の設定との互換)。"""
    if mode is True or mode == LOG10:
        return log_safe(X, 10)
    if mode == LOG2:
        return log_safe(X, 2)
    return X


def prep_matrix(df, compounds, max_missing, impute, log, zscore, index=None):
    """サンプル x 化合物 の数値行列を返す (欠損処理・対数・Z スコア)。"""
    X = df[compounds].astype(float)
    X.index = unique_names(df[LABEL_COL]) if index is None else index
    X = X.loc[:, X.notna().any() & (X.isna().mean() <= max_missing)]
    if impute == "最小値の1/2":
        X = X.fillna(X.min() / 2)
    elif impute == "0":
        X = X.fillna(0)
    X = apply_log(X, log)
    if zscore:
        X = zscore_cols(X)
    return X


# ---------------------------------------------------------------- 距離 (階層的クラスタリング)
DISTANCES = {
    "ユークリッド": "euclidean",
    "マンハッタン (cityblock)": "cityblock",
    "チェビシェフ": "chebyshev",
    "ミンコフスキー": "minkowski",
    "標準化ユークリッド": "seuclidean",
    "マハラノビス (Ledoit-Wolf 共分散)": "mahalanobis",
    "ピアソン相関 (1 - r)": "correlation",
    "スピアマン相関 (1 - ρ)": "spearman",
    "コサイン": "cosine",
    "キャンベラ": "canberra",
    "ブレイ・カーティス": "braycurtis",
}
# ユークリッド距離を前提とする連結法
EUCLIDEAN_ONLY = {"ward", "centroid", "median"}


def pairwise_distances(X, metric, p=3):
    """行どうしの距離を scipy の condensed 形式で返す。"""
    from scipy.spatial.distance import pdist
    from scipy.stats import rankdata

    X = np.asarray(X, dtype=float)
    if metric == "spearman":
        return pdist(np.apply_along_axis(rankdata, 1, X), "correlation")
    if metric == "mahalanobis":
        # サンプル数 < 次元数 だと標本共分散が特異になり、擬似逆行列では全ペアの距離がほぼ等しくなる。
        # Ledoit-Wolf の縮小推定で正則な共分散を使う。
        from sklearn.covariance import LedoitWolf

        return pdist(X, "mahalanobis", VI=np.linalg.inv(LedoitWolf().fit(X).covariance_))
    if metric == "minkowski":
        return pdist(X, "minkowski", p=p)
    return pdist(X, metric)


# ---------------------------------------------------------------- 代謝物の比
RATIO_SEP = " + "
# よく使われる比 (名前, 分子, 分母)。データに含まれる化合物だけ追加される
RATIO_PRESETS = [
    ("Kyn/Trp", ["Kynurenine"], ["Tryptophan"]),
    ("Phe/Tyr", ["Phenylalanine"], ["Tyrosine"]),
    ("Fischer 比 (BCAA/AAA)", ["Valine", "Leucine", "Isoleucine"], ["Phenylalanine", "Tyrosine"]),
    ("Cit/Arg", ["Citrulline"], ["Arginine"]),
    ("Arg/Orn", ["Arginine"], ["Ornitine"]),
    ("GABR (Arg/(Orn+Cit))", ["Arginine"], ["Ornitine", "Citrulline"]),
    ("Gln/Glu", ["Glutamine"], ["Glutamic acid"]),
    ("GSSG/GSH", ["Oxidized glutathione"], ["Glutathione"]),
    ("SAM/SAH", ["S-Adenosylmethionine"], ["S-Adenosylhomocysteine"]),
    ("Lac/Pyr", ["Lactic acid"], ["Pyruvic acid"]),
    ("3-HB/AcAc", ["3-ヒドロキシ酪酸"], ["アセト酢酸"]),
    ("AcCar/Car", ["Acetylcarnitine"], ["Carnitine"]),
]


def parse_terms(text):
    return [t.strip() for t in str(text or "").split("+") if t.strip()]


def validate_ratios(defs, compounds):
    """比の定義表 (名前, 分子, 分母) を検証し、[(名前, [分子], [分母])] とエラーの一覧を返す。"""
    ok, errors, names = [], [], set()
    for i, r in defs.iterrows():
        name, num, den = str(r.get("名前") or "").strip(), parse_terms(r.get("分子")), parse_terms(r.get("分母"))
        if not (name or num or den):
            continue
        name = name or f"{RATIO_SEP.join(num)} / {RATIO_SEP.join(den)}"
        missing = [t for t in num + den if t not in compounds]
        if not num or not den:
            errors.append(f"{name}: 分子と分母の両方が必要です")
        elif missing:
            errors.append(f"{name}: 化合物が見つかりません ({', '.join(missing)})")
        elif name in names or name in compounds:
            errors.append(f"{name}: 名前が重複しています")
        else:
            ok.append((name, num, den))
            names.add(name)
    return ok, errors


def add_ratios(base, ratios):
    """比の列を追加した表を返す。分子・分母のどれかが欠損 (未検出) なら NaN、分母が 0 でも NaN。"""
    out = base.copy()
    for name, num, den in ratios:
        n = out[num].astype(float).sum(axis=1, min_count=len(num))
        d = out[den].astype(float).sum(axis=1, min_count=len(den))
        out[name] = n / d.replace(0, np.nan)
    return out


# ---------------------------------------------------------------- 変数グループ (よく使う分類)
GROUP_SEP = "; "
GROUP_PRESETS = {
    "アミノ酸 (タンパク質構成)": ["Alanine", "Arginine", "Asparagine", "Aspartic acid", "Cysteine", "Glutamic acid",
                          "Glutamine", "Glycine", "Histidine", "Isoleucine", "Leucine", "Lysine", "Methionine",
                          "Phenylalanine", "Proline", "Serine", "Threonine", "Tryptophan", "Tyrosine", "Valine"],
    "分岐鎖アミノ酸 (BCAA)": ["Valine", "Leucine", "Isoleucine"],
    "アミノ酸関連 (非タンパク質性など)": ["Citrulline", "Ornitine", "4-Hydroxyproline", "b-alanine", "2-Aminobutyric acid",
                              "4-Aminobutyric acid", "Cystine", "Cystathionine", "Homocysteine", "Homocystine",
                              "Argininosuccinic acid", "Methionine sulfoxide", "Kynurenine", "Taurine", "Sarcosine"],
    "有機酸 (解糖・TCA 回路)": ["Pyruvic acid", "Lactic acid", "Citric acid", "Isocitric acid", "Aconitic acid",
                         "2-Ketoglutaric acid", "Succinic acid", "Fumaric acid", "Malic acid"],
    "核酸関連": ["Adenine", "Adenosine", "Adenosine monophosphate", "Guanine", "Guanosine", "Guanosine monophosphate",
             "Hypoxanthine", "Inosine", "IMP", "Xanthine", "Uric acid", "Cytosine", "Cytidine",
             "Cytidine monophosphate", "Uracil", "Uridine", "Thymine", "Thymidine", "Thymidine monophosphate",
             "Allantoin", "Orotic acid"],
    "メチル化・含硫": ["Methionine", "S-Adenosylmethionine", "S-Adenosylhomocysteine", "Homocysteine", "Cystathionine",
                "Cysteine", "Glutathione", "Oxidized glutathione", "Dimethylglycine", "Choline"],
    "アシルカルニチン・エネルギー": ["Carnitine", "Acetylcarnitine", "Creatine", "Creatinine", "3-ヒドロキシ酪酸", "アセト酢酸"],
}


def parse_members(text):
    return [t.strip() for t in str(text or "").split(";") if t.strip()]


def compound_stats(base, compounds, control=None):
    """化合物の並べ替えに使う指標: 平均・変動係数 CV% と、対照群との |log2FC| の最大値 (希釈グループ・condition ごと)。"""
    rows = {}
    for c in compounds:
        v = pd.to_numeric(base[c], errors="coerce")
        m = v.mean()
        cv = 100 * v.std() / m if m and pd.notna(m) else np.nan
        best = np.nan
        if control is not None and COND_COL in base:
            for _, g in base.groupby(GROUP_COL):
                means = g.groupby(COND_COL)[c].mean()
                ref = means.get(control, np.nan)
                for cond, mv in means.items():
                    if cond in (control, "") or not (ref > 0 and mv > 0):
                        continue
                    val = abs(np.log2(mv / ref))
                    best = val if np.isnan(best) else max(best, val)
        rows[c] = {"|log2FC|": best, "平均": m, "CV%": cv}
    return pd.DataFrame.from_dict(rows, orient="index")
