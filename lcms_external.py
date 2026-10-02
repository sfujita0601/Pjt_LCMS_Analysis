"""LC-MS 以外のデータ (脂質・臨床検査値など、サンプル ID ごとに複数の指標) の読み込みと統合。"""
import io
import re

import pandas as pd

MATCH_EXACT, MATCH_DIGITS = "完全一致 (空白・大文字小文字は無視)", "数字の部分だけで照合 (例: D-12 と 12)"


def read_table(raw_bytes, name, sheet=None):
    """CSV / TSV / Excel を読む。Excel はシート名を返すため (表, シート名の一覧) を返す。"""
    lower = name.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        book = pd.ExcelFile(io.BytesIO(raw_bytes))
        sheet = sheet if sheet in book.sheet_names else book.sheet_names[0]
        return book.parse(sheet), book.sheet_names
    for enc in ("utf-8-sig", "cp932"):
        try:
            text = raw_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("文字コードを判別できません (UTF-8 / Shift-JIS のみ対応)")
    sep = "\t" if lower.endswith((".tsv", ".txt")) or text.count("\t") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep), []


def numeric_columns(df, id_col, min_fraction=0.5):
    """数値として読める値が min_fraction 以上ある列 (ID 列を除く)。"""
    out = []
    for c in df.columns:
        if c == id_col:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() >= max(1, min_fraction * df[c].notna().sum()):
            out.append(c)
    return out


def match_key(value, mode, harmonize=False):
    s = str(value).strip()
    if s in ("", "nan", "None"):
        return None
    if mode == MATCH_DIGITS:
        nums = re.findall(r"\d+", s)
        return "-".join(str(int(n)) for n in nums) if nums else None
    s = re.sub(r"\s+", "", s).casefold()
    return s.replace("_", "-") if harmonize else s


def prepare(df, id_col, cols, prefix, mode, harmonize=False):
    """ID ごとに 1 行の数値表 (列名は「接頭辞: 列名」) にする。

    同じ ID が複数行ある場合は平均を取り、その数を返す。戻り値: (表 [index = 照合キー], 重複 ID 数)
    """
    d = df[[id_col] + list(cols)].copy()
    d["_key"] = d[id_col].map(lambda v: match_key(v, mode, harmonize))
    d = d.dropna(subset=["_key"])
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    dup = int(d["_key"].duplicated().sum())
    wide = d.groupby("_key")[list(cols)].mean()
    wide.columns = [f"{prefix}: {c}" if prefix else str(c) for c in cols]
    return wide, dup


def merge_into(base, label_col, wide, mode, harmonize=False):
    """解析用の表 (base) に外部データの列を label で結合する。(結合後の表, 一致した label, 一致しない外部 ID)"""
    keys = base[label_col].map(lambda v: match_key(v, mode, harmonize))
    joined = wide.reindex(keys.values)
    joined.index = base.index
    out = pd.concat([base, joined], axis=1)
    matched = sorted(set(base.loc[keys.isin(wide.index), label_col]))
    unmatched = sorted(set(wide.index) - set(keys.dropna()))
    return out, matched, unmatched
