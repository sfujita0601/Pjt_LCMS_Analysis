"""KEGG REST API (https://rest.kegg.jp) から経路図と座標 (KGML) を取得する。

KEGG REST API は学術利用向けに無償で提供されている。取得結果は .kegg_cache/ に保存し、同じ要求を繰り返さない。
文献: Kanehisa M, Goto S. Nucleic Acids Res. 2000;28:27-30. / Kanehisa M, et al. Nucleic Acids Res. 2023;51:D587-D592.
"""
import hashlib
import io
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
MAPPING_CSV = ROOT / "resources" / "kegg_compounds.csv"
CACHE_DIR = ROOT / ".kegg_cache"
BASE_URL = "https://rest.kegg.jp"
# 全体図 (01100 Metabolic pathways など) は図が巨大で座標の重ね描きに向かないので候補から除く
OVERVIEW_PREFIX = ("011", "012")
_last_request = [0.0]


def load_mapping():
    """化合物名 -> KEGG Compound ID の対応表 (resources/kegg_compounds.csv)。"""
    m = pd.read_csv(MAPPING_CSV, dtype=str, encoding="utf-8").fillna("")
    return dict(zip(m["compound"].str.strip(), m["kegg_id"].str.strip()))


def fetch(path, binary=False):
    """KEGG REST から取得する (ディスクキャッシュ付き、連続アクセスは 0.35 秒あける)。"""
    try:
        CACHE_DIR.mkdir(exist_ok=True)
    except OSError:
        pass
    key = path.strip("/").replace("/", "__").replace("+", "_")
    if len(key) > 120:  # 化合物 ID を並べた要求はファイル名が長くなるのでハッシュにする
        key = key[:40] + "_" + hashlib.sha1(key.encode()).hexdigest()
    cache = CACHE_DIR / (key + (".bin" if binary else ".txt"))
    if cache.exists():
        return cache.read_bytes() if binary else cache.read_text(encoding="utf-8")
    wait = 0.35 - (time.time() - _last_request[0])
    if wait > 0:
        time.sleep(wait)
    try:
        with urllib.request.urlopen(f"{BASE_URL}/{path.strip('/')}", timeout=30) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise FileNotFoundError(f"KEGG に見つかりません: {path}") from e
        raise
    finally:
        _last_request[0] = time.time()
    try:
        cache.write_bytes(data)
    except OSError:  # 書き込めない環境 (読み取り専用など) ではキャッシュしない
        pass
    return data if binary else data.decode("utf-8")


def pathways_for(compound_ids):
    """{経路 ID (map#####): [化合物 ID, ...]} を返す (全体図は除く)。"""
    ids = sorted({c for c in compound_ids if c})
    out = {}
    for i in range(0, len(ids), 50):
        text = fetch("link/pathway/" + "+".join(ids[i:i + 50]))
        for line in text.strip().splitlines():
            cpd, path = line.split("\t")
            pid = path.replace("path:", "")
            if pid.startswith("map") and not pid[3:].startswith(OVERVIEW_PREFIX):
                out.setdefault(pid, []).append(cpd.replace("cpd:", ""))
    return out


def pathway_names():
    text = fetch("list/pathway")
    return dict(line.replace("path:", "").split("\t", 1) for line in text.strip().splitlines())


def pathway_layout(map_id, org="map"):
    """経路図の画像 (PNG bytes) と化合物ノードの座標表を返す。

    参照経路 (map) の KGML は提供されていないので、座標は同じレイアウトの ko の KGML から取る。
    """
    num = map_id[-5:]
    image = fetch(f"get/{org}{num}/image", binary=True)
    kgml = fetch(f"get/{'ko' if org == 'map' else org}{num}/kgml")
    root = ET.fromstring(kgml)
    rows = []
    for e in root.iter("entry"):
        if e.get("type") != "compound":
            continue
        g = e.find("graphics")
        if g is None:
            continue
        for cid in e.get("name", "").split():
            rows.append({"kegg_id": cid.replace("cpd:", "").replace("gl:", ""), "x": float(g.get("x")),
                         "y": float(g.get("y")), "w": float(g.get("width", 8)), "h": float(g.get("height", 8))})
    return image, pd.DataFrame(rows, columns=["kegg_id", "x", "y", "w", "h"]), root.get("title", "")


def image_size(png_bytes):
    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as im:
        return im.size
