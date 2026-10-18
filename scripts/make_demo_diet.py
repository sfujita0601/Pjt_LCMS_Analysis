"""デモ用の飼料組成 (模擬データ) を作る。値は架空で、実在の飼料の組成ではない。

3 種類の飼料 (Ctrl / LowBCAA / HighArg) のアミノ酸含量 (g/kg 飼料) を書き出す。LowBCAA は分岐鎖アミノ酸を
半分に、HighArg はアルギニンを 2 倍にし、その分の窒素をグリシンで調整したという想定。

    uv run scripts/make_demo_diet.py                 # -> data/demo_diet_simulated.csv
    uv run scripts/make_demo_diet.py -o out.csv
"""
import argparse
from pathlib import Path

import pandas as pd

BASE = {  # g/kg 飼料 (架空)
    "Ala": 5.5, "Arg": 7.0, "Asp": 13.0, "Cys": 0.8, "Glu": 40.0, "Gly": 3.5, "His": 5.0, "Ile": 9.0, "Leu": 16.0,
    "Lys": 14.0, "Met": 5.0, "Phe": 9.0, "Pro": 19.0, "Ser": 9.5, "Thr": 7.5, "Trp": 2.2, "Tyr": 9.5, "Val": 11.0,
}


def make():
    ctrl = dict(BASE)
    low = dict(BASE, Ile=BASE["Ile"] / 2, Leu=BASE["Leu"] / 2, Val=BASE["Val"] / 2, Gly=BASE["Gly"] + 18.0)
    arg = dict(BASE, Arg=BASE["Arg"] * 2, Gly=BASE["Gly"] - 2.0)
    df = pd.DataFrame([ctrl, low, arg], index=["Ctrl", "LowBCAA", "HighArg"])
    df.index.name = "飼料"
    return df.round(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", default=str(Path(__file__).resolve().parent.parent / "data" / "demo_diet_simulated.csv"))
    args = ap.parse_args()
    make().to_csv(args.output, encoding="utf-8-sig")
    print(f"-> {args.output} (模擬データ)")


if __name__ == "__main__":
    main()
