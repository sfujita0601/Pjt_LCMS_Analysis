# LC-MS/MS 定量解析アプリ

島津 LabSolutions の定量結果 (テキスト出力) を読み込み、整形・QC・可視化・統計解析を行う Streamlit アプリです。

- 整形: 化合物ごとのブロックを「データファイル × 化合物」の濃度表に変換、内部標準 (IS) による補正
- QC: 検量線範囲での希釈の選択、希釈の直線性、STD の正確さ、LOD / LOQ、測定順のドリフト
- 前処理: サンプル間の正規化 (PQN・総量・中央値)、LOD / LOQ 未満の値のマスク、サンプル・化合物の選択
- 可視化: 棒グラフ、ヒートマップ、散布図、階層的クラスタリング、PCA、UMAP、相関ネットワーク、KEGG 経路図
- 統計: 2 群検定 (ボルケーノ)、3 群以上の比較と事後検定、代謝物の比
- 再現性: 解析設定を JSON に保存して読み込み直せる

各手法の説明と参考文献は、アプリ内の「解析手法の解説」ページにあります。

## 使い方 (手元の PC)

[uv](https://docs.astral.sh/uv/) が必要です。

```bash
uv sync                          # 初回のみ: 必要なライブラリを入れる
uv run streamlit run app.py      # ブラウザで http://localhost:8501 が開く
```

フォルダの名前や場所を変えたあとに `ModuleNotFoundError` や `bad interpreter` が出る場合は、仮想環境を作り直してください
(`.venv` の中のコマンドには作成時のフォルダの場所が書き込まれているため、`uv sync` だけでは直りません)。

```bash
rm -rf .venv && uv sync
```

起動は必ず `uv run streamlit run app.py` で行ってください。単に `streamlit run app.py` とすると、
別の環境の Streamlit が使われて同じエラーになることがあります。

コマンドラインで濃度表だけ作る場合:

```bash
uv run make_conc_table.py data/<ファイル名>.txt
```

## パスワードの設定

`.streamlit/secrets.toml.example` を `.streamlit/secrets.toml` にコピーしてパスワードを書きます。
`secrets.toml` が無ければ、手元の PC ではパスワードなしで起動します。

```toml
password = "共通のパスワード"

# 利用者ごとに分ける場合
[passwords]
yamada = "パスワード1"
```

## Streamlit Community Cloud で公開する

1. このフォルダを GitHub のリポジトリにする (測定データ `data/` とパスワード `secrets.toml` は `.gitignore` で除外済み)
2. https://share.streamlit.io で「Create app」→ リポジトリ・ブランチを選び、Main file path に `app.py` を指定
3. 「Advanced settings」で Python のバージョンを **3.12 以上** (既定の 3.12 のままで可) にし、Secrets に `password = "..."` を貼り付けて Deploy
4. Secrets を設定し忘れた場合、アプリは起動を止めてその旨を表示します (誰でも使える状態にはなりません)

ライブラリは `uv.lock` (または `requirements.txt`) から入ります。`pyproject.toml` を変えたときは、次のコマンドで
`requirements.txt` も更新してください。

```bash
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project > requirements.txt
```

### 注意

- アップロードしたデータはサーバーのメモリ上でだけ使われ、ブラウザを閉じると消えます。ただし Community Cloud は
  外部のサーバーなので、個人情報を含むデータや未発表の機密データを扱ってよいかは、所属機関のルールを確認してください
- KEGG の経路図は KEGG REST API (学術利用は無償) から取得します。経路図の画像を論文などに載せる場合は
  KEGG の利用条件 (https://www.kegg.jp/kegg/legal.html) を確認してください
- Community Cloud の無料枠はメモリが限られています。大きなデータや多数の同時利用では動作が遅くなることがあります
