"""レーダーチャート用の集計: 基準の condition に対する各群の評価値 (FC・差分・t 値など) とそのばらつき。

ばらつきは 下限・上限 の形で返す (FC を log2 値で計算した場合など、上下非対称になるため)。

比 (FC) のばらつきは誤差の伝搬則 (デルタ法) で求める。A, B を独立とすると
    FC = A / B,   SD(FC) ≈ FC · √((s_A / A)² + (s_B / B)²)
log2 値で計算する場合は d = mean(log2 A) − mean(log2 B), SD(d) = √(s_A² + s_B²) を求め、FC = 2^d, 範囲 = 2^(d ± k·SD(d))。
SEM は s / √n、95% 信頼区間は SEM に t 分布の分位点 (自由度は Welch-Satterthwaite の近似) を掛ける。
中央値を使う場合は、SD の代わりに頑健な SD (IQR / 1.349)、SEM の代わりに 1.2533 · 頑健な SD / √n (中央値の漸近標準誤差) を使う。
"""
import numpy as np
import pandas as pd
from scipy import stats

from lcms_analysis import COND_COL

M_VALUE = "値そのもの (濃度)"
M_FC = "Fold change (各群 / 基準)"
M_LOG2FC = "log2 Fold change"
M_PCT = "変化率 % ((各群 − 基準) / 基準)"
M_DIFF = "差分 (各群 − 基準)"
M_T = "t 値 (Welch)"
M_G = "効果量 Hedges の g"
M_Z = "Z スコア (基準群の平均・SD で標準化)"
METRICS = [M_LOG2FC, M_FC, M_PCT, M_DIFF, M_T, M_G, M_Z, M_VALUE]

C_MEAN, C_MEDIAN = "平均", "中央値"
D_NONE, D_SD, D_SEM, D_CI, D_BOOT = "なし", "SD (中央値のときは頑健な SD)", "SEM", "95% 信頼区間", "ブートストラップ 95% 区間"
DISPERSIONS = [D_SD, D_SEM, D_CI, D_BOOT, D_NONE]

S_NONE, S_MAX, S_Z = "そのまま", "化合物ごとに最大値を 1 にする", "化合物ごとに Z スコア化 (全群の試料の平均・SD)"


def baseline(metric):
    """基準群と同じ値になる位置 (図の基準線)。値そのものでは無し。"""
    return {M_FC: 1.0, M_LOG2FC: 0.0, M_PCT: 0.0, M_DIFF: 0.0, M_T: 0.0, M_G: 0.0, M_Z: 0.0}.get(metric)


def _center(x, center):
    return float(np.median(x)) if center == C_MEDIAN else float(np.mean(x))


def _spread(x, center):
    if len(x) < 2:
        return np.nan
    if center == C_MEDIAN:
        q1, q3 = np.percentile(x, [25, 75])
        return float((q3 - q1) / 1.349)
    return float(np.std(x, ddof=1))


def _se(x, center):
    s = _spread(x, center)
    return s * (1.2533 if center == C_MEDIAN else 1.0) / np.sqrt(len(x))


def _welch_df(u, v, na, nb):
    """u, v: 各群の (標準誤差)²。Welch-Satterthwaite の自由度。"""
    den = (u ** 2 / (na - 1) if na > 1 else 0) + (v ** 2 / (nb - 1) if nb > 1 else 0)
    return (u + v) ** 2 / den if den > 0 else max(na + nb - 2, 1)


def _k(disp, df):
    if disp == D_CI:
        return float(stats.t.ppf(0.975, df)) if np.isfinite(df) and df > 0 else 1.96
    return 1.0


def estimate(a, b, metric, center=C_MEAN, disp=D_SD, log2=False, ref_self=False):
    """1 化合物・1 群の (推定値, 下限, 上限, 備考)。a = 各群の値、b = 基準群の値 (log2 なら変換済み)。

    ref_self=True は基準群そのもの (a = b) を描くとき: 基準側のばらつきを 0 として、自身のばらつきだけを示す。
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or (metric != M_VALUE and nb == 0):
        return np.nan, np.nan, np.nan, "値なし"
    ca, sa, sea = _center(a, center), _spread(a, center), _se(a, center)
    if metric == M_VALUE:
        if disp == D_SD and center == C_MEDIAN:
            q1, q3 = np.percentile(a, [25, 75])
            return ca, float(q1), float(q3), "四分位範囲"
        h = {D_SD: sa, D_SEM: sea, D_CI: sea * _k(D_CI, na - 1)}.get(disp, np.nan)
        return ca, ca - h, ca + h, ""
    cb, sb, seb = _center(b, center), _spread(b, center), _se(b, center)
    if ref_self:
        sb_, seb_ = 0.0, 0.0
    else:
        sb_, seb_ = sb, seb
    use_se = disp in (D_SEM, D_CI)

    if metric in (M_FC, M_LOG2FC, M_PCT):
        if log2:  # 幾何平均の比 (中央値なら log2 値の中央値の差)
            d = ca - cb
            u, v = (sea ** 2, seb_ ** 2) if use_se else (sa ** 2, sb_ ** 2)
            h = np.sqrt(u + v) * _k(disp, _welch_df(sea ** 2, seb_ ** 2, na, nb))
            lo, hi = d - h, d + h
            if metric == M_LOG2FC:
                return d, lo, hi, ""
            fc, flo, fhi = 2.0 ** d, 2.0 ** lo, 2.0 ** hi
        else:
            if ca <= 0 or cb <= 0:
                return np.nan, np.nan, np.nan, "中心値が 0 以下で比を計算できない"
            fc = ca / cb
            u, v = ((sea / ca) ** 2, (seb_ / cb) ** 2) if use_se else ((sa / ca) ** 2, (sb_ / cb) ** 2)
            rel = np.sqrt(u + v) * _k(disp, _welch_df((sea / ca) ** 2, (seb_ / cb) ** 2, na, nb))
            if metric == M_LOG2FC:
                d, h = np.log2(fc), rel / np.log(2)
                return d, d - h, d + h, ""
            flo, fhi = fc * (1 - rel), fc * (1 + rel)
        if metric == M_FC:
            return fc, flo, fhi, ""
        return 100 * (fc - 1), 100 * (flo - 1), 100 * (fhi - 1), ""

    if metric == M_DIFF:
        d = ca - cb
        u, v = (sea ** 2, seb_ ** 2) if use_se else (sa ** 2, sb_ ** 2)
        h = np.sqrt(u + v) * _k(disp, _welch_df(sea ** 2, seb_ ** 2, na, nb))
        return d, d - h, d + h, ""

    if metric == M_T:
        if ref_self:
            return 0.0, np.nan, np.nan, ""
        ma, mb = a.mean(), b.mean()
        va, vb = (a.var(ddof=1) / na if na > 1 else np.nan), (b.var(ddof=1) / nb if nb > 1 else np.nan)
        t = (ma - mb) / np.sqrt(va + vb) if va + vb > 0 else np.nan
        return float(t), np.nan, np.nan, "t 値は平均で計算 (ばらつきは付けない)"

    if metric == M_G:
        if ref_self:
            return 0.0, np.nan, np.nan, ""
        if na < 2 or nb < 2:
            return np.nan, np.nan, np.nan, "n < 2"
        sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        g = (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (na + nb) - 9)) if sp > 0 else np.nan
        se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
        h = se * _k(disp, na + nb - 2)
        return float(g), g - h, g + h, "ばらつきは g の標準誤差 (SD・SEM とも)" if disp in (D_SD, D_SEM) else ""

    if metric == M_Z:
        if not sb or not np.isfinite(sb) or sb <= 0:
            return np.nan, np.nan, np.nan, "基準群の SD が 0 または計算できない"
        z = (ca - cb) / sb
        h = {D_SD: sa, D_SEM: sea, D_CI: sea * _k(D_CI, na - 1)}.get(disp, np.nan) / sb
        return z, z - h, z + h, ""
    raise ValueError(metric)


def bootstrap(a, b, metric, center, log2, ref_self=False, n_boot=2000, seed=0):
    """各群の中で復元抽出し、推定値の 2.5・97.5 パーセンタイルを返す。"""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, len(a))
        rb = b if (ref_self or metric == M_VALUE) else rng.choice(b, len(b))
        out[i] = estimate(ra, rb, metric, center, D_NONE, log2, ref_self)[0]
    out = out[np.isfinite(out)]
    if len(out) < n_boot / 2:
        return np.nan, np.nan
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _vals(df, cond, cpd, log2):
    x = pd.to_numeric(df.loc[df[COND_COL] == cond, cpd], errors="coerce").dropna().values.astype(float)
    if log2:
        x = np.log2(x[x > 0])
    return x


def radar_table(df, compounds, ref, conds, metric, center=C_MEAN, disp=D_SD, log2=False, include_ref=False,
                n_boot=2000, seed=0):
    """condition x 化合物 の long 表 (condition, 化合物, 値, 下限, 上限, n, n (基準), 備考)。"""
    rows = []
    show = ([ref] if include_ref and metric != M_VALUE else []) + [c for c in conds if c != ref or metric == M_VALUE]
    for cond in show:
        is_ref = cond == ref and metric != M_VALUE
        for cpd in compounds:
            a, b = _vals(df, cond, cpd, log2), _vals(df, ref, cpd, log2)
            if is_ref:
                a = b
            est, lo, hi, note = estimate(a, b, metric, center, D_NONE if disp == D_BOOT else disp, log2, is_ref)
            if disp == D_BOOT and len(a) and np.isfinite(est):
                lo, hi = bootstrap(a, b, metric, center, log2, is_ref, n_boot, seed)
                if len(a) < 5:
                    note = (note + "; " if note else "") + "n が少なくブートストラップ区間は不安定"
            rows.append({COND_COL: cond, "化合物": cpd, "値": est, "下限": lo, "上限": hi,
                         "n": len(a), "n (基準)": len(b) if metric != M_VALUE else np.nan, "備考": note})
    return pd.DataFrame(rows)


def rescale(table, mode, df=None, log2=False):
    """値そのもの のときの化合物ごとの尺度合わせ。下限・上限も同じ変換をする。"""
    t = table.copy()
    if mode == S_MAX:
        m = t.groupby("化合物")["値"].transform(lambda v: np.nanmax(np.abs(v)) if np.isfinite(v).any() else np.nan)
        for c in ("値", "下限", "上限"):
            t[c] = t[c] / m.replace(0, np.nan)
    elif mode == S_Z:
        conds = t[COND_COL].unique()
        sub = df[df[COND_COL].isin(conds)]
        for cpd, idx in t.groupby("化合物").groups.items():
            x = pd.to_numeric(sub[cpd], errors="coerce").dropna().values.astype(float)
            if log2:
                x = np.log2(x[x > 0])
            mu, sd = (x.mean(), x.std(ddof=1)) if len(x) > 1 else (np.nan, np.nan)
            for c in ("値", "下限", "上限"):
                t.loc[idx, c] = (t.loc[idx, c] - mu) / sd if sd > 0 else np.nan
    return t
