#!/usr/bin/env python3
"""LabSolutions の定量結果テキスト (化合物ごとのブロックが縦に連結された形式) を
行 = データファイル、列 = 化合物、値 = 濃度 の表 (CSV) に整形する。
データファイル名から label (サンプル名、希釈表記を除く) と 希釈倍率 (x10 など) の列も付ける。
さらに内部標準 (既定: 2-Morpholinoethanesulfonic acid) で補正した換算濃度表も出力する。

使い方:
    python make_conc_table.py data/20260925_Fujita.txt
    python make_conc_table.py data/20260925_Fujita.txt -o out.csv --na ND
"""
import argparse
import csv
import re
from pathlib import Path

FILE_COL = "データファイル名"
CONC_COL = "濃度"
LABEL_COL = "label"
DILUTION_COL = "希釈倍率"
RATIO_COL = "IS比率"
IS_NAME = "2-Morpholinoethanesulfonic acid"
MISSING = "-----"

# 20260925_<サンプル名>_<サンプルID>_<連番>.lcd  (標準はサンプルIDが空: 20260925_1uM STD A__003.lcd)
FNAME_RE = re.compile(r"^\d{8}_(?P<name>.*)_[^_]*_\d+\.lcd$")
DILUTION_RE = re.compile(r"\s*\bx(\d+)\b")


def parse_fname(fname):
    """データファイル名から (label, 希釈倍率) を返す。例: 'D-10 x10' -> ('D-10', 'x10')"""
    m = FNAME_RE.match(fname)
    if m:
        name = m.group("name")
    else:  # 想定外の形式: 日付直後の "_" から次の "_" まで
        name = fname.split("_")[1] if "_" in fname else fname
    d =DILUTION_RE.search(name)
    dilution = f"x{d.group(1)}" if d else ""
    label = DILUTION_RE.sub("", name).strip()
    return label, dilution


def decode_bytes(raw):
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    raise ValueError("文字コードを判別できません (UTF-8 / Shift-JIS のみ対応)")


def read_text(path):
    return decode_bytes(Path(path).read_bytes())


def parse_blocks(text):
    """{化合物名: {データファイル名: 濃度文字列}} と、出現順のファイル一覧を返す。"""
    blocks = {}
    files = []
    name = None
    header = None
    for line in text.splitlines():
        cells = line.split("\t")
        if cells[0] == "ID#":
            name, header = None, None
        elif cells[0] == "Name":
            name = cells[1].strip()
            if name in blocks:
                raise ValueError(f"化合物名が重複しています: {name}")
            blocks[name] = {}
        elif FILE_COL in cells:
            header = cells
        elif name is not None and header is not None and cells[0].strip().isdigit():
            row = dict(zip(header, cells))
            fname = row[FILE_COL].strip()
            blocks[name][fname] = row.get(CONC_COL, "").strip()
            if fname not in files:
                files.append(fname)
    return blocks, files


def is_std(label):
    return "STD" in label


def to_float(v):
    try:
        return float(v)
    except ValueError:
        return None


def normalize_by_is(files, blocks, meta, is_name):
    """内部標準で補正した濃度を返す。

    希釈倍率ごとに STD 以外のサンプルの IS 濃度の平均を取り、
    比率 = IS 濃度 / 平均、換算濃度 = 濃度 / 比率 とする。
    戻り値: [(データファイル名, 比率, {化合物: 換算濃度})]  (STD は除外)
    """
    samples = [f for f in files if not is_std(meta[f][0])]
    groups = {}
    for f in samples:
        is_val = to_float(blocks[is_name].get(f, ""))
        if is_val is not None:
            groups.setdefault(meta[f][1], []).append(is_val)
    means = {d: sum(v) / len(v) for d, v in groups.items()}
    for d, v in groups.items():
        print(f"IS 平均 [{d or '希釈なし'}]: {means[d]:.3f} (n={len(v)})")

    results = []
    for f in samples:
        is_val = to_float(blocks[is_name].get(f, ""))
        if not is_val:
            print(f"警告: {f} は IS 値が無いため補正できません")
            results.append((f, None, {}))
            continue
        ratio = is_val / means[meta[f][1]]
        corrected = {}
        for c in blocks:
            v = to_float(blocks[c].get(f, ""))
            if v is not None:
                corrected[c] = v / ratio
        results.append((f, ratio, corrected))
    return results


def write_csv(path, header, rows):
    # Excel で開いても文字化けしないよう BOM 付き UTF-8
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="LabSolutions からエクスポートした .txt")
    ap.add_argument("-o", "--output", help="出力 CSV (既定: 入力名_conc.csv)")
    ap.add_argument("--is-correct", action="store_true",
                    help="IS 補正した表も出力する。LabSolutions で外部標準法 (IS 補正なし) を使っている場合だけ指定してください "
                         "(内部標準法なら二重補正になります)")
    ap.add_argument("--is-output", help="IS 補正後の出力 CSV (既定: 入力名_conc_IS.csv)")
    ap.add_argument("--is-name", default=IS_NAME, help=f"内部標準の化合物名 (既定: {IS_NAME})")
    ap.add_argument("--na", default="", help=f"未検出 ({MISSING}) やデータ欠損を置き換える文字 (既定: 空欄)")
    ap.add_argument("--sort", action="store_true", help="データファイル名でソートする (既定: 入力の出現順)")
    args = ap.parse_args()

    blocks, files = parse_blocks(read_text(args.input))
    if args.sort:
        files.sort()
    compounds = list(blocks)
    meta = {f: parse_fname(f) for f in files}
    stem = Path(args.input).with_name(Path(args.input).stem)

    # 1) 生の濃度表
    out = Path(args.output) if args.output else Path(f"{stem}_conc.csv")
    rows = []
    for fname in files:
        vals = []
        for c in compounds:
            v = blocks[c].get(fname, "")
            vals.append(args.na if v in ("", MISSING) else v)
        rows.append([fname, *meta[fname]] + vals)
    write_csv(out, [FILE_COL, LABEL_COL, DILUTION_COL] + compounds, rows)
    print(f"{len(files)} files x {len(compounds)} compounds -> {out}")

    # 2) 内部標準で補正した換算濃度表 (STD を除く)。定量方式が分からないまま補正しないよう、指定した場合だけ
    if not args.is_correct:
        print("IS 補正した表は出力していません (外部標準法で補正が必要な場合は --is-correct を指定)")
        return
    if args.is_name not in blocks:
        raise SystemExit(f"内部標準 '{args.is_name}' がデータにありません")
    is_out = Path(args.is_output) if args.is_output else Path(f"{stem}_conc_IS.csv")
    rows = []
    for fname, ratio, corrected in normalize_by_is(files, blocks, meta, args.is_name):
        vals = [f"{corrected[c]:.3f}" if c in corrected else args.na for c in compounds]
        rows.append([fname, *meta[fname], "" if ratio is None else f"{ratio:.4f}"] + vals)
    write_csv(is_out, [FILE_COL, LABEL_COL, DILUTION_COL, RATIO_COL] + compounds, rows)
    print(f"{len(rows)} samples (STD 除く) -> {is_out}")


if __name__ == "__main__":
    main()
