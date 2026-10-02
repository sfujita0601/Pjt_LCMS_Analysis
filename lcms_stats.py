"""正規化・スケーリング・群間検定・相関解析。根拠となる文献は views/docs.py (解析手法の解説) を参照。"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from lcms_analysis import COND_COL, GROUP_COL, log2_safe
from make_conc_table import LABEL_COL, is_std

# ---------------------------------------------------------------- 正規化 (サンプルごとの補正)
NORM_NONE = "なし"
NORM_PQN = "PQN (確率的商正規化)"
NORM_SUM = "総量 (定数和)"
NORM_MEDIAN = "中央値"
NORMALIZATIONS = [NORM_NONE, NORM_PQN, NORM_SUM, NORM_MEDIAN]
REF_ALL, REF_CONTROL = "全サンプルの中央値", "対照群の中央値"


def normalization_factors(X, method, ref_mask=None, pre_total=True):
    """サンプル (行) ごとの希釈係数を返す。値をこの係数で割ると正規化される。

    係数の計算には、全サンプルで検出された化合物 (欠損のない列) だけを使う。
    総量・中央値は係数の中央値が 1 になるよう揃え、元の単位の大きさを保つ。
    """
    complete = X.loc[:, X.notna().all() & (X > 0).all()]
    if complete.shape[1] == 0:
        raise ValueError("全サンプルで検出された化合物が無いため正規化できません")
    if method == NORM_SUM:
        f = complete.sum(axis=1)
    elif method == NORM_MEDIAN:
        f = complete.median(axis=1)
    elif method == NORM_PQN:
        # Dieterle et al. (2006): (1) 総量正規化 → (2) 参照スペクトル (中央値) との商の中央値で割る
        Y = complete
        total = pd.Series(1.0, index=X.index)
        if pre_total:
            total = Y.sum(axis=1) / Y.sum(axis=1).median()
            Y = Y.div(total, axis=0)
        ref_rows = Y if ref_mask is None or not ref_mask.any() else Y[ref_mask]
        reference = ref_rows.median(axis=0)
        quotient = Y.div(reference, axis=1).median(axis=1)
        return (total * quotient).rename("係数")
    else:
        return pd.Series(1.0, index=X.index, name="係数")
    return (f / f.median()).rename("係数")


def normalize(base, compounds, method, ref=REF_ALL, control=None, pre_total=True):
    """希釈グループごとに正規化した表と、係数の表 (label, 希釈, 係数, 使用化合物数) を返す。STD は補正しない。"""
    if method == NORM_NONE:
        return base, pd.DataFrame()
    out = base.copy()
    logs = []
    for g, idx in out.groupby(GROUP_COL).groups.items():
        rows = [i for i in idx if not is_std(out.at[i, LABEL_COL])]
        if len(rows) < 2:
            continue
        X = out.loc[rows, compounds].astype(float)
        mask = (out.loc[rows, COND_COL] == control) if ref == REF_CONTROL and control else None
        f = normalization_factors(X, method, mask, pre_total)
        out.loc[rows, compounds] = X.div(f, axis=0)
        n_used = int((X.notna().all() & (X > 0).all()).sum())
        logs.append(pd.DataFrame({LABEL_COL: out.loc[rows, LABEL_COL], GROUP_COL: g, "係数": f.values,
                                  "使用化合物数": n_used}))
    return out, (pd.concat(logs, ignore_index=True) if logs else pd.DataFrame())


# ---------------------------------------------------------------- スケーリング (化合物ごと, 多変量解析の前処理)
SCALINGS = ["なし", "オート (Z スコア)", "パレート", "レンジ", "VAST", "レベル"]


def scale_columns(X, method):
    """van den Berg et al. (2006) のスケーリング。'なし' 以外は平均で中心化する。"""
    if method == "なし":
        return X
    mean, sd = X.mean(), X.std(ddof=1)
    keep = (sd > 0) & sd.notna()
    X, mean, sd = X.loc[:, keep], mean[keep], sd[keep]
    C = X - mean
    if method.startswith("オート"):
        return C / sd
    if method == "パレート":
        return C / np.sqrt(sd)
    if method == "レンジ":
        return C / (X.max() - X.min())
    if method == "VAST":
        return C / sd * (mean / sd)
    if method == "レベル":
        return C / mean
    raise ValueError(method)


# ---------------------------------------------------------------- 群間比較 (ボルケーノ)
TESTS = ["Welch の t 検定", "Student の t 検定", "Mann-Whitney U 検定", "Brunner-Munzel 検定"]
CORRECTIONS = {
    "Benjamini-Hochberg (FDR)": "fdr_bh",
    "Benjamini-Yekutieli (FDR)": "fdr_by",
    "二段階 Benjamini-Krieger-Yekutieli (FDR)": "fdr_tsbky",
    "Bonferroni": "bonferroni",
    "Holm": "holm",
    "Šidák": "sidak",
    "Holm-Šidák": "holm-sidak",
    "Hochberg": "simes-hochberg",
    "Hommel": "hommel",
    "補正なし": None,
}


def _test(a, b, test):
    with warnings.catch_warnings():  # 値がほぼ一定の群などで出る精度警告は p 値の解釈に注記で対応する
        warnings.simplefilter("ignore", RuntimeWarning)
        return _test_raw(a, b, test)


def _test_raw(a, b, test):
    if test.startswith("Welch"):
        return stats.ttest_ind(a, b, equal_var=False).pvalue
    if test.startswith("Student"):
        return stats.ttest_ind(a, b, equal_var=True).pvalue
    if test.startswith("Mann"):
        return stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    if test.startswith("Brunner"):
        return stats.brunnermunzel(a, b).pvalue
    raise ValueError(test)


def compare_groups(df, compounds, treat, control, test, correction, log_for_test=True, min_n=2):
    """treat vs control を化合物ごとに検定する。

    FC = treat の平均 / control の平均 (棒グラフ・ヒートマップの FC と同じ定義)。
    t 検定は log_for_test=True のとき log2 変換後の値に対して行う (濃度は右に裾を引くため)。
    片方の群で検出数が min_n 未満の化合物は p = NaN とし、多重性補正の対象から外す。
    """
    A, B = df[df[COND_COL] == treat], df[df[COND_COL] == control]
    rows = []
    for c in compounds:
        a, b = A[c].dropna().astype(float), B[c].dropna().astype(float)
        ma, mb = a.mean(), b.mean()
        fc = ma / mb if mb and pd.notna(mb) and mb != 0 else np.nan
        p = np.nan
        if len(a) >= min_n and len(b) >= min_n:
            ta, tb = a, b
            if log_for_test and "t 検定" in test:
                ta, tb = pd.Series(log2_safe(a)).dropna(), pd.Series(log2_safe(b)).dropna()
            if len(ta) >= min_n and len(tb) >= min_n and not (np.ptp(ta) == 0 and np.ptp(tb) == 0):
                with np.errstate(all="ignore"):
                    p = float(_test(ta, tb, test))
        rows.append({"化合物": c, "n (" + treat + ")": len(a), "n (" + control + ")": len(b),
                     "平均 (" + treat + ")": ma, "平均 (" + control + ")": mb, "FC": fc,
                     "log2FC": float(log2_safe([fc])[0]) if pd.notna(fc) else np.nan, "p": p})
    res = pd.DataFrame(rows)
    res["q"] = adjust_p(res["p"], CORRECTIONS[correction])
    return res


def adjust_p(p, method):
    p = pd.Series(p, dtype=float)
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if method is None:
        out[ok] = p[ok]
    elif ok.any():
        out[ok] = multipletests(p[ok], method=method)[1]
    return out


# ---------------------------------------------------------------- 相関
CORR_METHODS = {"Spearman": "spearman", "Pearson": "pearson", "Kendall": "kendall"}


def correlation(X, method, min_pairs=6):
    """化合物どうしの相関係数・p 値・ペア数 (欠損はペアごとに除外し、ペア数が min_pairs 未満なら NaN)。"""
    cols = X.columns
    n = len(cols)
    R = pd.DataFrame(np.eye(n), index=cols, columns=cols)
    P = pd.DataFrame(0.0, index=cols, columns=cols)
    N = pd.DataFrame(X.notna().sum().values[:, None].repeat(n, 1), index=cols, columns=cols)
    func = {"spearman": stats.spearmanr, "pearson": stats.pearsonr, "kendall": stats.kendalltau}[method]
    V = X.values
    for i in range(n):
        for j in range(i + 1, n):
            m = ~np.isnan(V[:, i]) & ~np.isnan(V[:, j])
            N.iat[i, j] = N.iat[j, i] = int(m.sum())
            if m.sum() < min_pairs or np.ptp(V[m, i]) == 0 or np.ptp(V[m, j]) == 0:
                r = p = np.nan
            else:
                res = func(V[m, i], V[m, j])
                r, p = float(res[0]), float(res[1])
            R.iat[i, j] = R.iat[j, i] = r
            P.iat[i, j] = P.iat[j, i] = p
    return R, P, N


def correlation_edges(R, P, N, r_min, q_max, correction="fdr_bh"):
    """|r| >= r_min かつ 補正後 p <= q_max の化合物ペア (上三角) を返す。"""
    iu = np.triu_indices(len(R), k=1)
    e = pd.DataFrame({"化合物1": R.index[iu[0]], "化合物2": R.columns[iu[1]],
                      "r": R.values[iu], "p": P.values[iu], "n": N.values[iu]}).dropna(subset=["r", "p"])
    e["q"] = adjust_p(e["p"], correction).values
    return e[(e["r"].abs() >= r_min) & (e["q"] <= q_max)].sort_values("r", key=np.abs, ascending=False)


# ---------------------------------------------------------------- 3 群以上の比較
ANOVA, WELCH_ANOVA, KRUSKAL = "一元配置 ANOVA", "Welch の ANOVA", "Kruskal-Wallis 検定"
MULTI_TESTS = [ANOVA, WELCH_ANOVA, KRUSKAL]
TUKEY, GAMES_HOWELL, DUNN, DUNNETT = "Tukey-Kramer (HSD)", "Games-Howell", "Dunn 検定", "Dunnett 検定"
PAIR_T, PAIR_WELCH, PAIR_MW = "Student の t 検定 + 補正", "Welch の t 検定 + 補正", "Mann-Whitney U 検定 + 補正"
PAIRS_ALL, PAIRS_CONTROL = "全ペア", "対照群との比較のみ"
POSTHOCS = {  # (全体の検定, 比較のしかた) -> 選べる事後検定
    PAIRS_ALL: {ANOVA: [TUKEY, PAIR_T], WELCH_ANOVA: [GAMES_HOWELL, PAIR_WELCH], KRUSKAL: [DUNN, PAIR_MW]},
    PAIRS_CONTROL: {ANOVA: [DUNNETT, PAIR_T], WELCH_ANOVA: [PAIR_WELCH], KRUSKAL: [DUNN, PAIR_MW]},
}
# 補正を選べる事後検定 (Tukey / Games-Howell / Dunnett は検定の分布で多重性を調整済み)
POSTHOC_NEEDS_ADJUST = {DUNN, PAIR_T, PAIR_WELCH, PAIR_MW}
PARAMETRIC = {ANOVA, WELCH_ANOVA}


def welch_anova(groups):
    """Welch (1951) の一元配置分散分析 (等分散を仮定しない)。(F, p) を返す。"""
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    m = np.array([np.mean(g) for g in groups])
    v = np.array([np.var(g, ddof=1) for g in groups])
    if (v <= 0).any():
        return np.nan, np.nan
    w = n / v
    W = w.sum()
    mw = (w * m).sum() / W
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    h = ((1 - w / W) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * h
    F = a / b
    df2 = (k ** 2 - 1) / (3 * h)
    return float(F), float(stats.f.sf(F, k - 1, df2))


def _omnibus(groups, test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if test == ANOVA:
            r = stats.f_oneway(*groups)
            return float(r.statistic), float(r.pvalue)
        if test == WELCH_ANOVA:
            return welch_anova(groups)
        r = stats.kruskal(*groups)
        return float(r.statistic), float(r.pvalue)


def _values(df, compound, conds, log):
    out = []
    for c in conds:
        v = df.loc[df[COND_COL] == c, compound].dropna().astype(float)
        if log:
            v = pd.Series(log2_safe(v)).dropna()
        out.append(v.values)
    return out


def compare_multi(df, compounds, conds, test, correction, log=True, min_n=2):
    """3 群以上の比較 (化合物ごとの全体検定)。log=True なら ANOVA 系は log2 値で行う。"""
    rows = []
    use_log = log and test != KRUSKAL
    for cpd in compounds:
        groups = _values(df, cpd, conds, use_log)
        row = {"化合物": cpd}
        for c, g in zip(conds, _values(df, cpd, conds, False)):
            row[f"n ({c})"] = len(g)
            row[f"平均 ({c})"] = g.mean() if len(g) else np.nan
        ok = [g for g in groups if len(g) >= min_n]
        stat, p = (np.nan, np.nan)
        if len(ok) == len(conds) and not all(np.ptp(g) == 0 for g in ok):
            stat, p = _omnibus(ok, test)
        row["統計量"], row["p"] = stat, p
        rows.append(row)
    res = pd.DataFrame(rows)
    res["q"] = adjust_p(res["p"], CORRECTIONS[correction])
    return res


def games_howell(groups, names):
    """Games & Howell (1976): 等分散を仮定しない多重比較 (スチューデント化範囲分布で調整)。"""
    k = len(groups)
    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = groups[i], groups[j]
            va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
            diff = np.mean(a) - np.mean(b)
            se = np.sqrt((va + vb) / 2)
            df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
            q = abs(diff) / se if se > 0 else np.nan
            p = float(stats.studentized_range.sf(q, k, df)) if np.isfinite(q) else np.nan
            rows.append((names[i], names[j], diff, p))
    return rows


def dunn(groups, names):
    """Dunn (1964): Kruskal-Wallis の後の順位和による多重比較 (同順位を補正した z 検定、補正前の p)。"""
    allv = np.concatenate(groups)
    ranks = stats.rankdata(allv)
    N = len(allv)
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum() / (12 * (N - 1))
    mean_r, start = [], 0
    for g in groups:
        mean_r.append(ranks[start:start + len(g)].mean())
        start += len(g)
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            se = np.sqrt((N * (N + 1) / 12 - tie) * (1 / len(groups[i]) + 1 / len(groups[j])))
            z = (mean_r[i] - mean_r[j]) / se
            rows.append((names[i], names[j], mean_r[i] - mean_r[j], float(2 * stats.norm.sf(abs(z)))))
    return rows


def posthoc(df, compound, conds, method, correction="Holm", log=True, min_n=2, control=None):
    """事後検定。control を指定すると 対照群 vs 各群 の比較だけを行う (群1 = 各群, 群2 = 対照群)。

    Tukey / Games-Howell / Dunnett は検定自体が多重性を調整した p、それ以外は correction で補正する
    (補正の対象は実際に行った比較の数: 全ペアなら k(k-1)/2、対照群との比較なら k-1)。
    """
    use_log = log and method not in (DUNN, PAIR_MW)
    vals = _values(df, compound, conds, use_log)
    keep = [i for i, g in enumerate(vals) if len(g) >= min_n]
    groups, names = [vals[i] for i in keep], [conds[i] for i in keep]
    if len(groups) < 2 or (control is not None and control not in names):
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if method == DUNNETT:
            ci = names.index(control)
            others = [i for i in range(len(groups)) if i != ci]
            r = stats.dunnett(*[groups[i] for i in others], control=groups[ci])
            rows = [(names[i], control, np.mean(groups[i]) - np.mean(groups[ci]), float(p))
                    for i, p in zip(others, r.pvalue)]
        elif method == TUKEY:
            r = stats.tukey_hsd(*groups)
            rows = [(names[i], names[j], np.mean(groups[i]) - np.mean(groups[j]), float(r.pvalue[i, j]))
                    for i in range(len(groups)) for j in range(i + 1, len(groups))]
        elif method == GAMES_HOWELL:
            rows = games_howell(groups, names)
        elif method == DUNN:
            rows = dunn(groups, names)
        else:
            test = {PAIR_T: "Student の t 検定", PAIR_WELCH: "Welch の t 検定", PAIR_MW: "Mann-Whitney U 検定"}[method]
            rows = []
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    p = _test_raw(groups[i], groups[j], test) if np.ptp(np.r_[groups[i], groups[j]]) > 0 else np.nan
                    rows.append((names[i], names[j], np.mean(groups[i]) - np.mean(groups[j]), float(p)))
    if control is not None and method != DUNNETT:
        # 対照群を含むペアだけ残し、向きを (各群 - 対照群) に揃える
        rows = [(a, b, d, p) if b == control else (b, a, -d, p) for a, b, d, p in rows if control in (a, b)]
    unit = "順位の差" if method == DUNN else ("log2 値の差" if use_log else "差")
    out = pd.DataFrame(rows, columns=["群1", "群2", f"{unit} (群1 - 群2)", "p"])
    if method in POSTHOC_NEEDS_ADJUST:
        out["補正後 p"] = adjust_p(out["p"], CORRECTIONS[correction]).values
    else:
        out["補正後 p"] = out["p"]
    return out


# ---------------------------------------------------------------- 正準相関分析 (CCA)
def _inv_sqrt(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-10, None)
    return (V * w ** -0.5) @ V.T


def _standardize(X, mean=None, sd=None):
    mean = X.mean(axis=0) if mean is None else mean
    sd = X.std(axis=0, ddof=1) if sd is None else sd
    return (X - mean) / sd, mean, sd


def cca_fit(X, Y, reg=0.0):
    """正準相関分析 (Hotelling, 1936)。reg > 0 なら共分散を単位行列へ縮小する正則化 CCA
    (canonical ridge; Vinod, 1976): C = (1 - reg) S + reg I。X, Y は標準化済み (n x p, n x q)。
    戻り値: (X の重み A [p x k], Y の重み B [q x k])
    """
    n, p = X.shape
    q = Y.shape[1]
    Cxx = (1 - reg) * (X.T @ X) / (n - 1) + reg * np.eye(p)
    Cyy = (1 - reg) * (Y.T @ Y) / (n - 1) + reg * np.eye(q)
    Cxy = (1 - reg) * (X.T @ Y) / (n - 1)
    Kx, Ky = _inv_sqrt(Cxx), _inv_sqrt(Cyy)
    U, _, Vt = np.linalg.svd(Kx @ Cxy @ Ky, full_matrices=False)
    k = min(p, q, n - 1)
    return Kx @ U[:, :k], Ky @ Vt.T[:, :k]


def _score_corr(U, V):
    return np.array([np.corrcoef(U[:, i], V[:, i])[0, 1] for i in range(U.shape[1])])


def cca(X, Y, reg=0.0, n_perm=999, seed=0, loo=True):
    """CCA の結果をまとめて返す。

    - 正準相関 (各成分の正準変量どうしの相関)
    - 並べ替え検定: Y のサンプルの対応をランダムに入れ替えて同じ計算を繰り返し、各成分の正準相関が
      偶然これ以上になる割合を p 値とする (サンプル数が少ない・変数が多い場合でも使える)
    - 一つ抜き交差検証 (LOO): 1 サンプルを除いて重みを求め、除いたサンプルの正準変量を予測して相関を取る。
      変数がサンプル数に比べて多いと、当てはめの正準相関は過大になるので、汎化の目安にする
    - 構造相関 (各変数と正準変量の相関) と交差構造相関 (各変数と相手側の正準変量の相関)
    """
    Xs, _, _ = _standardize(X)
    Ys, _, _ = _standardize(Y)
    A, B = cca_fit(Xs, Ys, reg)
    U, V = Xs @ A, Ys @ B
    r = _score_corr(U, V)
    sign = np.sign(r)  # 正準相関を正にそろえる
    B, V, r = B * sign, V * sign, np.abs(r)
    k = len(r)

    pvals = np.full(k, np.nan)
    if n_perm:
        rng = np.random.default_rng(seed)
        count = np.zeros(k)
        for _ in range(n_perm):
            Yp = Ys[rng.permutation(len(Ys))]
            Ap, Bp = cca_fit(Xs, Yp, reg)
            count += np.abs(_score_corr(Xs @ Ap, Yp @ Bp)) >= r - 1e-12
        pvals = (count + 1) / (n_perm + 1)

    r_loo = np.nan
    if loo and len(X) >= 5:
        u_out, v_out = [], []
        for i in range(len(X)):
            tr = np.arange(len(X)) != i
            Xt, mx, sx = _standardize(X[tr])
            Yt, my, sy = _standardize(Y[tr])
            a, b = cca_fit(Xt, Yt, reg)
            a, b = a[:, 0], b[:, 0]
            # 学習データ内で正準相関が正になるよう b の向きをそろえる (除いたサンプルの情報は使わない)。
            # a と b を同時に反転しても予測点は原点に対して反転するだけなので、相関は変わらない
            b = b * np.sign(np.corrcoef(Xt @ a, Yt @ b)[0, 1])
            u_out.append(((X[i] - mx) / sx) @ a)
            v_out.append(((Y[i] - my) / sy) @ b)
        r_loo = float(np.corrcoef(u_out, v_out)[0, 1])

    def corr_with(M, S):
        return np.array([[np.corrcoef(M[:, j], S[:, c])[0, 1] for c in range(S.shape[1])] for j in range(M.shape[1])])

    return dict(r=r, p=pvals, r_loo=r_loo, U=U, V=V, A=A, B=B,
                load_x=corr_with(Xs, U), load_y=corr_with(Ys, V),
                cross_x=corr_with(Xs, V), cross_y=corr_with(Ys, U))
