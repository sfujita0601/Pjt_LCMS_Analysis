"""飼料のアミノ酸組成と、血中アミノ酸の変化の比較。

飼料組成の表 (行 = 飼料、列 = アミノ酸、値 = 含量) を読み込み、condition と飼料を対応づけて、
アミノ酸ごとに「飼料中の変化 (log2 比、対照飼料に対する)」と「血中の変化 (log2 幾何平均比と 95% 信頼区間)」を並べる。
"""
import re

import numpy as np
import pandas as pd

# アミノ酸の別名 (3 文字略号・英名・日本語名など) -> 代表名 (小文字)。データ中の化合物名とはこの代表名で照合する
_ALIASES = {
    "alanine": ["ala", "a", "alanine", "l-alanine", "アラニン"],
    "arginine": ["arg", "r", "arginine", "l-arginine", "アルギニン"],
    "asparagine": ["asn", "n", "asparagine", "l-asparagine", "アスパラギン"],
    "aspartic acid": ["asp", "d", "aspartic acid", "aspartate", "l-aspartic acid", "アスパラギン酸"],
    "cysteine": ["cys", "c", "cysteine", "l-cysteine", "システイン"],
    "cystine": ["cystine", "cys2", "(cys)2", "l-cystine", "シスチン"],
    "glutamine": ["gln", "q", "glutamine", "l-glutamine", "グルタミン"],
    "glutamic acid": ["glu", "e", "glutamic acid", "glutamate", "l-glutamic acid", "グルタミン酸"],
    "glycine": ["gly", "g", "glycine", "グリシン"],
    "histidine": ["his", "h", "histidine", "l-histidine", "ヒスチジン"],
    "isoleucine": ["ile", "i", "isoleucine", "l-isoleucine", "イソロイシン"],
    "leucine": ["leu", "l", "leucine", "l-leucine", "ロイシン"],
    "lysine": ["lys", "k", "lysine", "l-lysine", "リジン", "リシン"],
    "methionine": ["met", "m", "methionine", "l-methionine", "メチオニン"],
    "phenylalanine": ["phe", "f", "phenylalanine", "l-phenylalanine", "フェニルアラニン"],
    "proline": ["pro", "p", "proline", "l-proline", "プロリン"],
    "serine": ["ser", "s", "serine", "l-serine", "セリン"],
    "threonine": ["thr", "t", "threonine", "l-threonine", "トレオニン", "スレオニン"],
    "tryptophan": ["trp", "w", "tryptophan", "l-tryptophan", "トリプトファン"],
    "tyrosine": ["tyr", "y", "tyrosine", "l-tyrosine", "チロシン"],
    "valine": ["val", "v", "valine", "l-valine", "バリン"],
    "ornithine": ["orn", "ornithine", "ornitine", "オルニチン"],
    "citrulline": ["cit", "citrulline", "シトルリン"],
    "taurine": ["tau", "taurine", "タウリン"],
    "hydroxyproline": ["hyp", "hydroxyproline", "4-hydroxyproline", "ヒドロキシプロリン"],
}
_CANON = {alias: canon for canon, names in _ALIASES.items() for alias in names}


def _norm(name):
    s = re.sub(r"\s*\(.*?\)\s*", " ", str(name)).strip().lower()  # 単位などの括弧を除く
    s = re.sub(r"\s+", " ", s)
    return _CANON.get(s, s)


def match_compounds(diet_columns, compounds):
    """飼料の列名 -> データ中の化合物名 (別名で照合。見つからなければ "")。"""
    by_canon = {}
    for c in compounds:
        by_canon.setdefault(_norm(c), c)
    return {col: by_canon.get(_norm(col), "") for col in diet_columns}


def read_diet(raw_bytes, name):
    """飼料組成の表を読む (CSV / TSV / Excel)。1 列目を飼料名とし、残りの数値の列をアミノ酸とする。"""
    import lcms_external as lx

    df, _ = lx.read_table(raw_bytes, name)
    df = df.dropna(how="all")
    name_col = df.columns[0]
    df[name_col] = df[name_col].astype(str).str.strip()
    num = lx.numeric_columns(df, name_col)
    out = df.set_index(name_col)[num].apply(pd.to_numeric, errors="coerce")
    out.index.name = "飼料"
    return out


def diet_log2_ratio(diet, condition_to_diet, control):
    """condition ごとに log2 (その飼料の含量 / 対照の飼料の含量)。対照の含量が 0 または欠損なら NaN。"""
    ref = condition_to_diet.get(control)
    if ref not in diet.index:
        return pd.DataFrame()
    rows = {}
    for cond, dname in condition_to_diet.items():
        if cond == control or dname not in diet.index:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log2(diet.loc[dname] / diet.loc[ref])
        rows[cond] = r.replace([np.inf, -np.inf], np.nan)
    return pd.DataFrame(rows).T


def combine(diet_ratio, plasma, mapping, q_thr=0.05, diet_thr=0.25):
    """飼料の変化と血中の変化を アミノ酸 x condition の long 表にまとめ、組み合わせを分類する。

    plasma: compare_groups の結果を condition ごとに縦に連結したもの (列 比較群 を持つ)
    mapping: 飼料の列名 -> 化合物名
    分類 (探索の目安): 飼料と同じ向きに有意 / 飼料と逆向きに有意 / 飼料は変化したが血中は有意差なし /
    飼料はほぼ同じだが血中が有意に変化 / どちらも変化なし
    """
    rows = []
    for cond in diet_ratio.index:
        p = plasma[plasma["比較群"] == cond].set_index("化合物")
        for col, cpd in mapping.items():
            if not cpd or cpd not in p.index or col not in diet_ratio.columns:
                continue
            dx = diet_ratio.at[cond, col]
            r = p.loc[cpd]
            dy, q = r["log2 幾何平均比"], r["q"]
            sig = pd.notna(q) and q <= q_thr
            changed = pd.notna(dx) and abs(dx) >= diet_thr
            if sig and changed:
                cls = "飼料と同じ向きに有意に変化" if np.sign(dx) == np.sign(dy) else "飼料と逆向きに有意に変化"
            elif changed:
                cls = "飼料は変化したが血中は有意差なし"
            elif sig:
                cls = "飼料はほぼ同じだが血中が有意に変化"
            else:
                cls = "どちらも変化なし"
            rows.append({"condition": cond, "飼料の列": col, "化合物": cpd, "飼料 log2 比": dx,
                         "血中 log2 比": dy, "95% CI 下限": r["95% CI 下限"], "95% CI 上限": r["95% CI 上限"],
                         "p": r["p"], "q": q, "分類": cls})
    return pd.DataFrame(rows)


def association(table):
    """condition ごとに、アミノ酸をまたいだ 飼料 log2 比 と 血中 log2 比 の相関 (探索用)。"""
    from scipy import stats

    rows = []
    for cond, g in table.groupby("condition"):
        g = g.dropna(subset=["飼料 log2 比", "血中 log2 比"])
        n = len(g)
        if n < 3 or g["飼料 log2 比"].nunique() < 2:
            rows.append({"condition": cond, "アミノ酸の数": n})
            continue
        rs, ps = stats.spearmanr(g["飼料 log2 比"], g["血中 log2 比"])
        rp, pp = stats.pearsonr(g["飼料 log2 比"], g["血中 log2 比"])
        slope = np.polyfit(g["飼料 log2 比"], g["血中 log2 比"], 1)[0]
        rows.append({"condition": cond, "アミノ酸の数": n, "Spearman ρ": rs, "p (Spearman)": ps,
                     "Pearson r": rp, "p (Pearson)": pp, "回帰の傾き": slope})
    return pd.DataFrame(rows)
