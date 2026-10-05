"""画面の部品のキー (key=...) が重複しないことを、ソースコードから確認する。

共通部品 (compound_picker / prep_controls など) は、渡した名前に接尾辞を付けてキーを作る。各タブで直接書いたキーや、
別の共通部品が作るキーと同じ名前になると StreamlitDuplicateElementKey で落ちるため、組み合わせを確認する。
接尾辞は、関数の中の key=f"{key}..." と、key をそのまま渡している別の関数の呼び出しから自動で集める。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "views" / "analysis.py", *sorted(ROOT.glob("ui_*.py"))]
SOURCE = "\n".join(p.read_text(encoding="utf-8") for p in FILES)


def function_bodies():
    bodies = {}
    for m in re.finditer(r"^def (\w+)\(.*?(?=^def |^# =|^[A-Za-z_]+ = |\Z)", SOURCE, re.S | re.M):
        bodies[m.group(1)] = m.group(0)
    return bodies


BODIES = function_bodies()


def suffixes(func, seen=()):
    """関数 func に名前 key を渡したときに作られるキーの接尾辞。"""
    body = BODIES.get(func, "")
    out = set(re.findall(r'key=f"\{key\}([^"{}]*)"', body))
    if re.search(r"key=key\b", body):
        out.add("")
    for callee in BODIES:
        if callee != func and callee not in seen and re.search(rf"\b{callee}\([^)\n]*\bkey\b", body):
            out |= suffixes(callee, seen + (func,))
    return out


def generated_keys():
    """各タブで共通部品を呼んだ名前 x その部品の接尾辞。"""
    keys = []
    for func in BODIES:
        sfx = suffixes(func)
        if not sfx:
            continue
        for name in re.findall(rf'\b{func}\("([^"]+)"', SOURCE):
            keys += [name + s for s in sorted(sfx)]
    return keys


def fixed_keys():
    """f 文字列を使わずに直接書かれたキー。"""
    return re.findall(r'key="([^"{}]+)"', SOURCE)


def test_fixed_keys_are_unique():
    keys = fixed_keys()
    dup = sorted({k for k in keys if keys.count(k) > 1})
    assert not dup, f"同じキーが複数あります: {dup}"


def test_generated_keys_do_not_collide():
    gen = generated_keys()
    assert gen, "共通部品のキーを見つけられません (テストの前提が崩れています)"
    clash = sorted(set(gen) & set(fixed_keys()))
    assert not clash, f"共通部品が作るキーと、直接書いたキーが重なっています: {clash}"
    dup = sorted({k for k in gen if gen.count(k) > 1})
    assert not dup, f"共通部品が作るキーどうしが重なっています: {dup}"
