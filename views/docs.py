"""解析手法の解説ページ (正規化を中心に、各機能の根拠となる文献を示す)。"""
import streamlit as st

st.title("解析手法の解説")
st.caption("このアプリが行う処理の内容と、その根拠となる文献をまとめています。番号 [n] は末尾の参考文献に対応します。")

st.markdown("""
## 処理の流れ

1. **LabSolutions の濃度を読み込む** — 未検出 (-----) と数値として読めない値 (入力エラー) を区別する
2. **値の状態を判定** — LOD / LOQ (測定ごと)・検量範囲から、採用可能 / 確認必要 / LOD 未満 / LOQ 未満 / 検量上限超過 /
   検量下限未満 / 未検出 / 測定なし / 入力エラー に分ける
3. **測定の採用** — 希釈を統合する場合は検量範囲に基づいて採用。採用できる測定が無ければ **再測定候補** (値は採用しない)
4. **IS 係数** — 定量方式を「外部標準法」と設定し、補正を有効にした場合のみ
5. **希釈係数** — 希釈測定の倍率が LabSolutions で未適用と設定した場合のみ
6. **体積換算係数** — 設定した場合のみ (最終値は「元の血清・血漿中濃度」)
7. **マスク** — LOD / LOQ 未満の値の置き換え (設定した場合のみ。既定はしない)
8. **サンプルの選択と技術反復の平均** — 試料以外 (QC・ブランク) を除き、同じ個体ID・時点の行を平均 (生物学的 n を正しく数える)
9. **サンプル間の正規化** (任意、既定はなし) — PQN・総量・中央値
10. **対数変換・スケーリング** (各解析の前処理) — log10、オート / パレート / レンジ / VAST / レベル

エクスポートに含まれない情報 (定量方式・希釈係数の適用状態・体積換算・単位) は推測で補わず、サイドバーで設定します。
既定は、利用者の測定法 (外部標準法・LabSolutions では希釈倍率を未適用・単位 µM) に合わせ、IS 補正あり・換算濃度 (ラット) です。
未設定を選んだ項目は補正・換算を行いません。
値ごとの経過は「処理履歴・状態」タブで確認・出力できます。

**絶対濃度と相対プロファイルを分けてください。** 血清・血漿の濃度を比べる解析は 1〜8 までの値 (正規化なし) で行います。
9 の正規化と 10 のスケーリングは相対的なプロファイルを探索するための設定で、正規化後の値は濃度 (µM など) ではありません
(図や表では「正規化値」と表示します)。
""")

st.markdown("""
## 1. 内部標準による補正と二重補正の防止

LabSolutions で **内部標準法** (IS 面積比の検量線) を使っている場合、濃度は既に IS で補正されています。
アプリで同じ補正をすると二重補正になるため、定量方式を「内部標準法」と設定するとアプリの IS 補正は選べません。
定量方式が **未設定** のあいだも補正しません。

**外部標準法** (面積の検量線) の場合に限り、注入量やイオン化の変動を補う目的で、アプリで IS 補正を行えます。
参考として、データから「濃度が面積に比例するか (外部標準法)、面積 / IS 面積 に比例するか (内部標準法)」を診断して表示しますが、
検量線の切片や重み付けの影響を受けるため、最終的な判断は LabSolutions のメソッドで確認してください。

アプリの IS 補正は次の式です (希釈グループ内で計算)。
""")
st.latex(r"r_i = \frac{\mathrm{IS}_i}{\overline{\mathrm{IS}}}, \qquad x'_{ij} = \frac{x_{ij}}{r_i}")
st.markdown("""
$x_{ij}$ はサンプル $i$ の化合物 $j$ の濃度、$\\overline{\\mathrm{IS}}$ は STD を除くサンプルの IS 濃度の平均です。

1 種類の IS で全化合物を補正すると、IS と性質 (保持時間・イオン化のしかた) が大きく異なる化合物では補正が不十分、
あるいは過剰になることがあります。複数の IS から化合物ごとに最適な組み合わせを選ぶ方法 (NOMIS) も提案されています [1]。
""")

st.markdown("""
## 1b. 換算濃度 (バイアル中濃度 → 元の血清中濃度)

LabSolutions の濃度は、外部標準法の検量線で求めた **バイアル (測定液 20 µL) 中の濃度** です。
前処理の体積比を掛けて、元の血清中の濃度 (換算濃度) にします。統計には換算濃度を使います (既定)。
""")
st.latex(r"c_{\mathrm{血清}} = c_{\mathrm{バイアル}} \times 200 \times \frac{380}{280} \times \frac{170}{130} \times \frac{1}{V_{\mathrm{血清}}}"
         r"\;(\times\ \text{希釈倍率})")
st.markdown("""
- $V_{\\mathrm{血清}}$ は出発血清量 (µL)。ラットは 50 µL (係数 7.0989)、マウスは血清 25 µL + 水 25 µL で 25 µL (係数 14.1978)
- 試料ごとに出発血清量が違う場合は、条件設定 タブの「血清量 (µL)」で上書きします
- x10 は 10 倍希釈を意味し、LabSolutions では倍率を掛けていないため、換算濃度にするときに 10 倍します (既定)。
  「希釈測定の倍率」を未設定にすると倍率を掛けず、処理履歴の「最終値の意味」に「換算濃度ではない」と表示します
- 換算係数は全試料で一定 (または試料ごとに決まった値) なので、群間の比 (FC) や順位に基づく検定の結果は変わりませんが、
  濃度の絶対値・差・平均 ± SD の値は換算後の単位になります

## 2. 検量線の範囲と希釈測定

定量値が信頼できるのは、検量線を作った濃度範囲 (定量下限 LLOQ 〜 定量上限 ULOQ) の中だけです。
生体試料分析法バリデーションのガイドラインでは、ULOQ を超えた試料は希釈して再測定し、
希釈しても正確さ・精度が保たれること (dilution integrity) を確認するよう求めています [2, 3]。

本アプリの既定「自動 (化合物単位)」では、希釈なしの濃度が 1 サンプルでも上限を超えた化合物は、全サンプルで希釈測定の値を使います。
同じ化合物の中で希釈なしと希釈測定を混ぜると、両者の間の系統差 (マトリックス効果・非直線性・測定時刻の違い) が
サンプル間の見かけの差になってしまうためです。両者の一致は 検量線・希釈 タブの「希釈の直線性」で確認できます。

「自動 (サンプル単位)」では、化合物ごと・サンプルごとに次のように選びます。

| 希釈なしの濃度 | 採用する値 |
|---|---|
| 上限以下 | 希釈なしの値 (下限未満なら「下限未満」として記録) |
| 上限を超える | 希釈測定の値 × 希釈倍率 (希釈測定が無ければ希釈なしの値を「上限超過」として記録) |
| 未検出 | 希釈測定があればその値 × 希釈倍率 |

範囲の判定には LabSolutions の算出濃度 (測定液中濃度) を使います。

**採用のルール (統合する場合)**: 優先する測定が検量範囲内 (または範囲未評価) ならそれを採用し、上限を超えていれば
もう一方の測定を確認します。どちらも採用できなければ **値を採用せず「再測定候補」** とします (元の値は処理履歴に残ります)。
両方の希釈が範囲内のときは、希釈測定 x 倍率 / 希釈なし が許容 (既定 ±20%) を外れると「確認必要」にします。
同じ試料・同じ希釈の再注入は、既定では平均せず最初の測定を採用します (設定で平均も選べます)。

1 点だけで検量している化合物は範囲を評価できないため「確認必要 (検量範囲 未評価)」とします。
検量線・希釈タブで下限・上限を編集すると、その範囲で判定します。
""")

st.markdown("""
## 3. サンプル間の正規化

**血清・血漿の濃度を比べる場合は、サンプル間の正規化をしないのが既定です。** 血液は採取量を揃えられ、
濃度そのものに意味があるためです。以下の方法は、相対的なプロファイルを探索する場合の任意の設定です。

細胞数・組織重量・尿の濃縮度など、**サンプル全体の「濃さ」の違い** (サンプルごとの希釈) を補正します。
いずれの方法も「大部分の化合物は群間で変化しない」ことを前提にしています。多くの化合物が一方向に変化する実験では、
正規化が本当の変化を打ち消してしまうことがあるので注意してください。

本アプリでは、正規化係数の計算に **そのグループの全サンプルで検出された化合物だけ** を使い、
内部標準と STD は計算・補正の対象から外しています。係数はテーブルタブで確認できます。

### PQN (Probabilistic Quotient Normalization, 確率的商正規化) [4]

Dieterle らが NMR メタボノミクスで尿の希釈の違いを補正するために提案した方法です。
「サンプル全体が一律に希釈されていれば、各化合物の 参照値に対する比 (商) は同じ値に集まる」という考えに基づきます。
""")
st.latex(r"""
\begin{aligned}
&\text{(1) 総量正規化:}\quad \tilde{x}_{ij} = x_{ij} \Big/ \frac{\sum_j x_{ij}}{\operatorname{median}_i \sum_j x_{ij}} \\
&\text{(2) 参照:}\quad m_j = \operatorname{median}_{i \in \text{参照群}}\ \tilde{x}_{ij} \\
&\text{(3) 希釈係数:}\quad d_i = \operatorname{median}_j \frac{\tilde{x}_{ij}}{m_j} \\
&\text{(4) 正規化:}\quad x^{\text{PQN}}_{ij} = \tilde{x}_{ij} / d_i
\end{aligned}
""")
st.markdown("""
- 原著では (1) の総量正規化を先に行い、参照には全サンプル (または対照群) の中央値スペクトルを使います [4]。
  本アプリでも両方を選べます (サイドバー「PQN の参照」「PQN の前に総量正規化」)。
- 商の **中央値** を使うので、一部の化合物が大きく変化しても係数が引っ張られにくいのが利点です。
- PQN を含む複数の正規化法を同じデータで比べた研究もあり、方法によって結果が変わりうることが示されています [5]。
  データの性質に合わせて、複数の方法で結論が変わらないか確認するのが安全です。

### 総量正規化 (constant sum / total)

各サンプルの濃度の合計が等しくなるように揃えます (本アプリでは係数の中央値が 1 になるよう調整)。
計算は単純ですが、一部の高濃度化合物 (乳酸・尿素など) が合計の大部分を占めるターゲット分析では、
その化合物の変化がすべての化合物の補正に持ち込まれます。また合計を一定にする操作は、
化合物間に見かけの負の相関を生みます (組成データのクロージャー問題)。PQN の比較対象として使うのがおすすめです。

### 中央値正規化

各サンプルの濃度の中央値で割ります。総量より外れ値に強い一方、測定化合物が少ないと係数が不安定になります。
""")

st.markdown("""
## 4. 対数変換とスケーリング (クラスタリング・PCA・UMAP)

濃度は桁の違う化合物が混在し、分布も右に裾を引くので、多変量解析の前に対数変換とスケーリングを行います。
van den Berg らはメタボロミクスデータに対する中心化・スケーリング・変換の効果を系統的に比較しています [6]。
$\\bar{x}_j$ は化合物 $j$ の平均、$s_j$ は標準偏差です。

| 方法 | 式 | 特徴 [6] |
|---|---|---|
| オート (Z スコア) | $(x - \\bar{x}_j) / s_j$ | すべての化合物を同じ重みで扱う。測定誤差の大きい低濃度の化合物も強調される |
| パレート | $(x - \\bar{x}_j) / \\sqrt{s_j}$ | 大きな変動をある程度残しつつ、桁の差を縮める |
| レンジ | $(x - \\bar{x}_j) / (\\max_j - \\min_j)$ | 生物学的な変動幅に対して揃える。外れ値に弱い |
| VAST | $\\dfrac{x - \\bar{x}_j}{s_j} \\cdot \\dfrac{\\bar{x}_j}{s_j}$ | 変動係数の小さい (安定な) 化合物を重視する |
| レベル | $(x - \\bar{x}_j) / \\bar{x}_j$ | 平均に対する相対変化。値の大きさに意味がある場合に向く |

本アプリの既定は log10 変換 + オートスケーリングです。欠損値は化合物ごとの最小値の 1/2 で補完しています。
""")

st.markdown("""
## 5. Fold Change と群間検定 (ボルケーノプロット)

**Fold Change (FC)** = 比較群の平均 / 対照群の平均 です (棒グラフ・ヒートマップ・ボルケーノで共通)。
検定結果の表には、実際に使った n、検定名、比較対象、効果量 (log2 値の平均の差 = log2 幾何平均比 とその 95% 信頼区間 (Welch)、
Hedges の g)、p 値、補正後 p 値を出力します。同じ個体を 2 条件で測った場合は、個体ID で対応づけた対応のある検定
(対応のある t 検定・Wilcoxon の符号付き順位検定) を選べます。多群比較では効果量 η² (Kruskal-Wallis は ε²)、
二元配置では偏 η² と係数の 95% 信頼区間を出力します。混合効果モデルは未実装です。

| 検定 | 前提・特徴 |
|---|---|
| Welch の t 検定 [7] | 等分散を仮定しない t 検定。2 群の比較で最初に選ぶのが一般的 |
| Student の t 検定 | 等分散を仮定する |
| Mann-Whitney U 検定 [8] | 順位に基づくノンパラメトリック検定。n が小さいと取りうる p 値の最小値が大きい |
| Brunner-Munzel 検定 [9] | 分散の違いにも頑健な順位検定 |

t 検定は既定で log2 変換した値に対して行います (濃度の分布は右に裾を引くため)。

### 多重性補正

多数の化合物を同時に検定すると、偶然 p < 0.05 になる化合物が増えます。

| 方法 | 制御する誤り | 特徴 |
|---|---|---|
| Bonferroni | FWER (1 つでも偽陽性を出す確率) | 最も保守的。p × 検定数 |
| Šidák [10] | FWER | 検定が独立なら Bonferroni よりわずかに強力 |
| Holm [11] | FWER | Bonferroni を段階的にしたもので、常に Bonferroni 以上の検出力 |
| Holm-Šidák | FWER | Holm の Šidák 版 |
| Hochberg [12] | FWER | 検定間が独立か正の相関のとき有効。Holm より強力 |
| Hommel [13] | FWER | Hochberg よりさらに強力 (同じ前提) |
| Benjamini-Hochberg [14] | FDR (有意とした中の偽陽性の割合) | 探索的なメタボロミクスで最もよく使われる |
| Benjamini-Yekutieli [15] | FDR | 検定間に任意の依存があっても成り立つが保守的 |
| 二段階 BKY [16] | FDR | 真の帰無仮説の割合を推定して BH より強力にしたもの |

代謝物どうしは相関していることが多いので、探索目的では FDR (BH) が、確認目的では FWER (Holm) が使われることが多いです。
""")

st.markdown("""
### 3 群以上の比較 (多群比較タブ)

| 全体の検定 | 前提 | 事後検定 |
|---|---|---|
| 一元配置 ANOVA | 正規分布・等分散 | Tukey-Kramer (HSD) [22, 23] |
| Welch の ANOVA [24] | 正規分布 (等分散は仮定しない) | Games-Howell [25] |
| Kruskal-Wallis 検定 [26] | 分布を仮定しない (順位) | Dunn 検定 [27] |

事後検定は「全ペア」と「対照群との比較のみ」を選べます。対照群との比較だけが目的なら、比較の数が
k(k−1)/2 から k−1 に減るぶん検出力が上がります。ANOVA の後の対照群との比較には Dunnett 検定 [30] を使います
(Welch・Kruskal-Wallis の後は、2 群検定または Dunn 検定を対照群との比較に限って多重性補正します)。

全体の検定の p 値には化合物間の多重性補正をかけます。事後検定のうち Tukey-Kramer と Games-Howell は
スチューデント化範囲分布で群の組み合わせの多重性を調整済みで、Dunn 検定と 2 群検定の繰り返しには
Holm などの補正をかけます。全体の検定が有意でない化合物の事後検定は、探索的な参考値として扱ってください。

### 二元配置分散分析 (二元配置 ANOVA タブ)

2 つの要因 (例: 食餌 × 薬剤) とその交互作用を、化合物ごとに次のモデルで検定します。
""")
st.latex(r"y = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon")
st.markdown("""
- **交互作用** $(\alpha\beta)_{ij}$ は「一方の要因の効果が、もう一方の要因の水準によって変わるか」を表します (交互作用の図で線が平行でない)
- **平方和の種類**: 各組み合わせのサンプル数が同じ (釣り合い型) なら Type II と Type III は一致します。釣り合っていない場合、
  交互作用が無ければ Type II の方が検出力が高く、交互作用がある場合の主効果の解釈は慎重に行う必要があります [36]
- **回帰係数** (ヒートマップの既定) は処理コーディングで求めます。主効果の係数は「もう一方の要因が基準水準のときの、
  基準水準との差」、交互作用の係数は「差の差」です。log2 変換していれば、それぞれ log2 FC とその差になります
- 全化合物を同時に検定するので、要因 (行) ごとに化合物間の多重性補正をかけます

### 検定と一緒に載せる図

慣習的には、平均を比べる検定 (t 検定・ANOVA) には 平均 ± SD (または SEM) の棒グラフ、
順位に基づく検定 (Mann-Whitney・Kruskal-Wallis) には中央値と四分位を示す箱ひげ図が使われます。
「図の種類: 自動」はこの対応で選びます。ただし棒グラフは分布の形や外れ値を隠してしまうため、
特にサンプル数が少ないときは、各サンプルの点を重ねるか、点 + 平均 ± SD の図が推奨されています [31]。
SEM は「平均の推定の精度」を表し、データのばらつきを表すのは SD です。

## 6. STD の正確さ

FDA と ICH M10 のガイドラインでは、検量線の標準試料の正確さ (算出濃度 / 設定濃度) を
**100 ± 15% (LLOQ は ± 20%)** 以内とし、標準の **75% 以上 (少なくとも 6 濃度)** がこれを満たすことを求めています [2, 3]。
本アプリはこの基準で各 STD と化合物を判定します。1 点検量の化合物は正確さが定義上 100% になるので評価できません。

### 検量線の情報と確認済み定量下限

LabSolutions のエクスポートには検量線の式・重み付け・切片が含まれないため「未取得」とします。
R² が高いことだけでは定量の妥当性を判断できないため、各 STD の逆算誤差 (正確さ) と、繰り返し測定の精度 (CV) を示します。
**確認済み LLOQ** は、繰り返し測定 (n ≥ 2) があり、偏り ±20% 以内かつ CV 20% 以下の最も低い濃度です [2, 3]。
LabSolutions が S/N = 10 から算出した LOQ は **推定 LOQ** で、正確さ・精度を実測で確認した下限ではありません。

### QC 試料の評価

条件設定タブの「試料種別」で、既知濃度 QC・プール QC・溶媒ブランク・処理ブランク・キャリーオーバー確認ブランクを区別します。

- **既知濃度 QC** (独立に調製した既知濃度の試料): 理論濃度に対する偏りと精度 (CV)
- **プール QC** (試料を混ぜたもの、真値は不明): 精度 (CV) と測定順のドリフト。真値が無いので絶対濃度の正確さは判定しません
- **キャリーオーバー確認ブランク**: 確認済み LLOQ に対する割合 (目安 20% 以下 [2, 3])

QC が無い項目は「未評価」とし、合格扱いにしません。閾値は化合物ごとに設定・保存できます。

### 定量限界・検出限界

LabSolutions は測定ごとにノイズを求め、S/N = 3 となる濃度を検出限界 (LOD)、S/N = 10 となる濃度を
定量限界 (LOQ) として出力しています (S/N に基づく方法 [28])。LOQ 未満の値は「検出はされたが定量値の信頼性が低い」値です。

マスクの方法は 3 通りから選べます。

- **欠損にする**: 最も保守的。ただし低濃度側の値だけが消えるので、平均が高い方に偏ります
- **限界値の 1/2 に置き換え**: 「検出限界未満は半分の値」とする簡便法で広く使われます
- **限界値に置き換え**: 上限側の推定

低濃度側で欠損になった値 (左側打ち切り) は、ランダムな欠損とは性質が異なり、補完の方法によって
統計解析の結果が変わることが報告されています [29]。結論がマスクの方法で変わらないか確認するのが安全です。

## 7. 測定順のドリフト

LC-MS では、測定が進むにつれてイオン源の汚れなどで感度が変わることがあります (ドリフト)。
大規模な測定では、全サンプルを混ぜたプール QC 試料を一定間隔で測定し、その変化から補正します (QC-RLSC など) [17]。
QC 試料の使い方と許容基準は Broadhurst らのガイドラインにまとめられています [18]。
本アプリでは、QC が無い場合の代わりとして、内部標準・STD・サンプルの値と測定順の Spearman 相関、
回帰直線の傾き、CV% を示します。condition と測定順が偏っていると、ドリフトと群間差を区別できません。

**トレンドの表示とドリフト補正は分けています。** 一般試料のトレンドから補正すると群差まで消すおそれがあるため、
アプリはドリフト補正を自動では行いません。QC に基づく補正 (QC-RLSC など) には測定順に沿って配置したプール QC が必要で、
現在のデータでは実装していません。

## 8. 相関解析

Pearson (直線関係)、Spearman・Kendall (単調な関係、外れ値に頑健) を選べます。p 値には多重性補正をかけ、
|r| と補正後 p の両方の閾値を満たすペアをネットワークの辺にします。代謝物間の相関は、共通の酵素や
調節機構だけでなく、系全体の変動からも生じるため、相関は直接の反応関係を意味するとは限りません [19]。
サンプル数が少ないと偶然大きな相関が出やすいので、ペアごとの最小サンプル数を設けています。

## 9. 代謝物の比

代謝物の比 (例: Kyn/Trp は IDO の活性、Fischer 比 (分岐鎖アミノ酸 / 芳香族アミノ酸) は肝機能、
GSSG/GSH は酸化ストレスの指標として用いられる) は、サンプルごとの希釈や正規化の係数が分子と分母で打ち消し合うため、
正規化の方法に依存しません。比の分布は右に裾を引くことが多いので、検定では対数 (log2) を取るのが一般的です。

## 10. 外部データと正準相関分析

### 外部データの統合
脂質・臨床検査値など、サンプル ID ごとに複数の指標を持つ表 (CSV / TSV / Excel) を読み込み、label と照合して統合します。
外部データの指標は IS 補正・サンプル間正規化の対象外です (単位や測定原理が異なるため)。対数変換は、負の値を
含まない列にだけ行います。

### 正準相関分析 (CCA)
2 つの変数グループ X (p 変数) と Y (q 変数) について、相関が最大になる線形結合 $u = Xa$, $v = Yb$ を順に求めます [32]。
""")
st.latex(r"\rho_1 = \max_{a, b} \operatorname{corr}(Xa, Yb), \qquad \text{以降は前の成分と無相関という条件の下で最大化}")
st.markdown("""
サンプル数 n に対して p + q が大きいと、標本共分散行列が特異になり正準相関は自明に 1 になります。
本アプリでは共分散を単位行列へ縮小する正則化 (canonical ridge) [33, 34] を使えます。
""")
st.latex(r"C_{xx}(\lambda) = (1 - \lambda) S_{xx} + \lambda I, \quad C_{yy}(\lambda) = (1 - \lambda) S_{yy} + \lambda I")
st.markdown("""
CCA は **探索** に使います。結果は「学習データ内の探索結果」であり、独立に検証された関連や因果関係ではありません。
小標本で学習データ内の正準相関が高くなることを理由に解析を禁止はしませんが、次を区別して表示します。

- **個別相関**: X の各変数 x Y の各変数の相関 (Pearson / Spearman)、有効 n・欠測数・補正後 p。生データの相関で、群や共変量は考慮していません
- **正準係数** ($a$, $b$): 線形合成を作る重み。多重共線性で不安定になりやすく、大きさで変数を順位付けしません
- **負荷量**: 変数と自分の側の正準変量の相関。**解釈はまず負荷量で行います**
- **交差負荷量**: 変数と相手側の正準変量の相関
- **正準スコア**: 各試料の正準変量の値
- **同じモードに寄与する組み合わせ**: 探索スコア = X 側の負荷量 x Y 側の負荷量 (第 1 モード)。個別の相関係数や p 値ではありません

**診断と安定性**: 完全ケース数・独立した生物学的単位 (個体) の数・変数の数・行列の階数・条件数、除いた変数 (欠測率・定数列) と
試料を表示します。個体を 1 つずつ除いて再解析し、負荷量の範囲・符号の反転・上位の変数に入る割合を示します。各回の負荷量は
全データの負荷量に符号を合わせ、一致 |r| が 0.7 未満の回は「対応するモードが不明確」とします。これは探索結果の安定性で、予測性能ではありません。

**任意の参考情報**: 交差検証 (個体単位の一つ抜き) は、補完・標準化・重みの推定を訓練データ内で行います。
置換検定 [35] は、反復測定 (同じ個体の複数時点) がある場合は行いません。「condition 内で置換」を選ぶと、群間差を保ったまま群内の関連を検定します。
正規分布を仮定する Bartlett の近似検定 (Wilks の Λ) は n が p + q より十分大きいときにしか使えないため採用していません。

模擬データ (架空の値) を含む場合は、画面・図・出力に「模擬データ」と表示します。

## 11. 再現性 (解析設定の保存)

サイドバーの「解析設定を保存」で、アプリの版 (git のコミット)・処理の順序・乱数の種・除外の履歴 (使わなかった label・除いた状態)・
適用した係数 (測定ごとの IS・希釈・体積係数) と、画面のすべての設定 (データの扱い・正規化・マスク・条件設定・サンプルの使用/除外・
検量線の選択・比の定義・各タブの設定・KEGG の対応表) と、データファイルの SHA-256 ハッシュ、主要ライブラリの
バージョンを JSON に保存します。同じデータファイルを選んでから「解析設定を読み込む」で復元すると、
同じ結果が得られます。ハッシュが一致しない場合は警告を表示します。
「解析結果をまとめて出力」では、最終濃度表・状態・処理履歴・QC 評価・検量線の情報・統計結果・設定と実行情報を zip で出力します
(図は各図の「図を保存」で個別に保存します)。外部データもファイル名とハッシュを記録します
(data/ に置いたファイルは名前で選び直せます)。UMAP と相関ネットワークの配置は乱数の種を固定しています。

## 12. KEGG PATHWAY

KEGG [20, 21] の経路図と、図上の化合物の座標 (KGML) を KEGG REST API から取得し、
測定した化合物の位置に log2FC の色を重ねます。化合物名と KEGG Compound ID の対応は
`resources/kegg_compounds.csv` にあり、KEGG で名称を照合済みです。KEGG の画像を論文などに掲載する場合は
利用条件 (https://www.kegg.jp/kegg/legal.html) を確認してください。
""")

st.markdown("""
## 参考文献

1. Sysi-Aho M, Katajamaa M, Yetukuri L, Orešič M. Normalization method for metabolomics data using optimal selection of multiple internal standards. *BMC Bioinformatics*. 2007;8:93. doi:10.1186/1471-2105-8-93
2. U.S. Food and Drug Administration. *Bioanalytical Method Validation: Guidance for Industry*. May 2018.
3. International Council for Harmonisation. *ICH M10: Bioanalytical Method Validation and Study Sample Analysis*. 2022.
4. Dieterle F, Ross A, Schlotterbeck G, Senn H. Probabilistic quotient normalization as robust method to account for dilution of complex biological mixtures. Application in ¹H NMR metabonomics. *Anal Chem*. 2006;78(13):4281–4290. doi:10.1021/ac051632c
5. Kohl SM, Klein MS, Hochrein J, Oefner PJ, Spang R, Gronwald W. State-of-the art data normalization methods improve NMR-based metabolomic analysis. *Metabolomics*. 2012;8(Suppl 1):146–160. doi:10.1007/s11306-011-0350-z
6. van den Berg RA, Hoefsloot HCJ, Westerhuis JA, Smilde AK, van der Werf MJ. Centering, scaling, and transformations: improving the biological information content of metabolomics data. *BMC Genomics*. 2006;7:142. doi:10.1186/1471-2164-7-142
7. Welch BL. The generalization of "Student's" problem when several different population variances are involved. *Biometrika*. 1947;34(1–2):28–35.
8. Mann HB, Whitney DR. On a test of whether one of two random variables is stochastically larger than the other. *Ann Math Stat*. 1947;18(1):50–60.
9. Brunner E, Munzel U. The nonparametric Behrens-Fisher problem: asymptotic theory and a small-sample approximation. *Biometrical J*. 2000;42(1):17–25.
10. Šidák Z. Rectangular confidence regions for the means of multivariate normal distributions. *J Am Stat Assoc*. 1967;62(318):626–633.
11. Holm S. A simple sequentially rejective multiple test procedure. *Scand J Stat*. 1979;6(2):65–70.
12. Hochberg Y. A sharper Bonferroni procedure for multiple tests of significance. *Biometrika*. 1988;75(4):800–802.
13. Hommel G. A stagewise rejective multiple test procedure based on a modified Bonferroni test. *Biometrika*. 1988;75(2):383–386.
14. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. *J R Stat Soc B*. 1995;57(1):289–300.
15. Benjamini Y, Yekutieli D. The control of the false discovery rate in multiple testing under dependency. *Ann Stat*. 2001;29(4):1165–1188.
16. Benjamini Y, Krieger AM, Yekutieli D. Adaptive linear step-up procedures that control the false discovery rate. *Biometrika*. 2006;93(3):491–507.
17. Dunn WB, Broadhurst D, Begley P, et al. Procedures for large-scale metabolic profiling of serum and plasma using gas chromatography and liquid chromatography coupled to mass spectrometry. *Nat Protoc*. 2011;6(7):1060–1083. doi:10.1038/nprot.2011.335
18. Broadhurst D, Goodacre R, Reinke SN, et al. Guidelines and considerations for the use of system suitability and quality control samples in mass spectrometry assays applied in untargeted clinical metabolomic studies. *Metabolomics*. 2018;14(6):72. doi:10.1007/s11306-018-1367-3
19. Camacho D, de la Fuente A, Mendes P. The origin of correlations in metabolomics data. *Metabolomics*. 2005;1(1):53–63. doi:10.1007/s11306-005-1107-3
20. Kanehisa M, Goto S. KEGG: Kyoto Encyclopedia of Genes and Genomes. *Nucleic Acids Res*. 2000;28(1):27–30. doi:10.1093/nar/28.1.27
21. Kanehisa M, Furumichi M, Sato Y, Kawashima M, Ishiguro-Watanabe M. KEGG for taxonomy-based analysis of pathways and genomes. *Nucleic Acids Res*. 2023;51(D1):D587–D592. doi:10.1093/nar/gkac963
22. Tukey JW. Comparing individual means in the analysis of variance. *Biometrics*. 1949;5(2):99–114.
23. Kramer CY. Extension of multiple range tests to group means with unequal numbers of replications. *Biometrics*. 1956;12(3):307–310.
24. Welch BL. On the comparison of several mean values: an alternative approach. *Biometrika*. 1951;38(3–4):330–336.
25. Games PA, Howell JF. Pairwise multiple comparison procedures with unequal n's and/or variances: a Monte Carlo study. *J Educ Stat*. 1976;1(2):113–125.
26. Kruskal WH, Wallis WA. Use of ranks in one-criterion variance analysis. *J Am Stat Assoc*. 1952;47(260):583–621.
27. Dunn OJ. Multiple comparisons using rank sums. *Technometrics*. 1964;6(3):241–252.
28. International Council for Harmonisation. *ICH Q2(R2): Validation of Analytical Procedures*. 2023.
29. Wei R, Wang J, Su M, et al. Missing value imputation approach for mass spectrometry-based metabolomics data. *Sci Rep*. 2018;8:663. doi:10.1038/s41598-017-19120-0
30. Dunnett CW. A multiple comparison procedure for comparing several treatments with a control. *J Am Stat Assoc*. 1955;50(272):1096–1121.
31. Weissgerber TL, Milic NM, Winham SJ, Garovic VD. Beyond bar and line graphs: time for a new data presentation paradigm. *PLoS Biol*. 2015;13(4):e1002128. doi:10.1371/journal.pbio.1002128
32. Hotelling H. Relations between two sets of variates. *Biometrika*. 1936;28(3–4):321–377.
33. Vinod HD. Canonical ridge and econometrics of joint production. *J Econom*. 1976;4(2):147–166.
34. González I, Déjean S, Martin PGP, Baccini A. CCA: an R package to extend canonical correlation analysis. *J Stat Softw*. 2008;23(12):1–14.
35. Winkler AM, Renaud O, Smith SM, Nichols TE. Permutation inference for canonical correlation analysis. *NeuroImage*. 2020;220:117065. doi:10.1016/j.neuroimage.2020.117065
36. Langsrud Ø. ANOVA for unbalanced data: use Type II instead of Type III sums of squares. *Stat Comput*. 2003;13(2):163–167.
""")
