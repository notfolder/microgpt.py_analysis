# microgpt.py_analysis

microgpt.pyを色々解析してみたい.

## モデルの保存と推論

json形式でモデルを保存するようにした
- microgpt_modelsave.patch

保存したモデルで推論できるようにした
- generate.py

引数:
- モデルファイル名: --model / -m: モデルファイル名を指定（デフォルト: model.json）
- 学習データファイル名: --input / -i: 学習データファイル名を指定（デフォルト: input.txt）
- ランダムシード: --seed / -s: ランダムシードを指定（デフォルト: 42）
- 出力ファイル名: --output / -o: 出力ファイル名を指定（デフォルト: generated_new_samples.txt

## モデルの可視化システム

学習済みmicroGPTモデルの内部構造と推論挙動を可視化するシステムを実装しました。

### セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# モデルの学習（model.jsonが生成されます）
python microgpt.py

# Jupyter Notebookの起動
cd visualize
jupyter notebook
```

### 可視化Notebook一覧

1. **v1_model_structure.ipynb**: モデル構造とパラメータ統計
2. **v2_attention_heatmap.ipynb**: Attention重みのヒートマップ
3. **v3_embedding_space.ipynb**: Embedding空間の分布（音韻的特徴＋Python記号分類）
4. **v4_inference_animation.ipynb**: 推論プロセスの可視化
5. **v5_probability_distribution.ipynb**: トークン間の条件付き確率分布

詳細は[visualization_spec.md](visualization_spec.md)を参照してください。

## 俳句モデルの作成

青空文庫から俳句を抽出したaozora_haiku_30000.txt(約4000句)を学習させた.  
microgpt.pyは26文字の英字のみを想定しているため、日本語を学習させるには最低限パラメータ調整しないとまともな俳句が出にくい.  
まともな俳句を生成するにはmicrogpt.pyの範疇を超えるモデルを作る必要がある.

- 学習済みモデル: haiku_orgparam_model.json
- 生成結果: generated_haiku_samples01.txt

俳句生成
``` bash
% python generate.py -m haiku_orgparam_model.json -i aozora_haiku_30000.txt -o generated_haiku_samples01.txt
```

なんとも微妙なラインナップ。意味不明な文言も並ぶ
```
来の草のうたりや雨の花
鶯の間に鐘に返のなる
山の夜の底の淋えるゝ哉
秋のなりつてゝやなり哉
```

## 16文字のpythonプログラム生成

16文字のpythonプログラムを学習、生成できるか検証したい。オリジナルが3万の名前でやっているので、頑張って3万パターンの16文字以下のpythonプログラムを生成するシステムを考える.  
LM Studioを使用して、意味のある16文字以下のPythonワンライナーを30,000パターン生成し、テキストファイルに出力するシステムの仕様を作成.



- python_oneliner_spec.md

モデルはdeepseek-coder-v2-lite-instruct-mlxを利用。6000パターンの生成を目指す.   
6000パターンの根拠は下記。

最低でも 1万〜5万トークン（文字）、安定学習には 5万〜20万トークン程度が必要。
これはモデル規模（約4k〜10kパラメータ）と 16文字系列の情報量から導かれる。

⸻

1. 前提整理

想定：
 - パラメータ数 $P \approx 4{,}000$
 - 文字語彙 $V \approx 80$
 - 生成長 T = 16
 - 文字レベル自己回帰モデル

⸻

2. 情報理論的下限

(1) 条件付きエントロピー

Python文字列の条件付きエントロピーは概算：

$H(X_t \mid X_{<t}) \approx 1.5 \sim 2.5 \text{ bits}$

16文字全体の情報量：

$H_{\text{seq}} \approx 16 \times 2 = 32 \text{ bits}$

1サンプルあたり約32ビットの情報。

⸻

(2) パラメータ推定の必要情報量

統計推定理論では、

$N \cdot I \gtrsim P$

が経験則となる。

ここで：
 - N：サンプル数
 - I：1サンプルあたりの情報量（bits）
 - P：自由度（パラメータ数）

よって：

$N \gtrsim \frac{P}{H_{\text{seq}}}$

$N \gtrsim \frac{4000}{32} \approx 125$

これは理論的最小値（極端に楽観的）。

⸻

3. 実用的サンプル数（汎化を考慮）

実際には：
 - 勾配ノイズ
 - 非凸最適化
 - 汎化誤差
 - パラメータ相関

があるため、

経験則として：

$N_{\text{tokens}} \approx 10P \sim 50P$

が妥当。

P=4000 の場合：

$40,000 \sim 200,000 \text{ tokens}$

⸻

4. 16文字系列に換算

1サンプル = 16文字とすると：

$\frac{40,000}{16} = 2,500$

$\frac{200,000}{16} = 12,500$

つまり：

約2,500〜12,000個の16文字コード例

⸻

5. 交差エントロピー収束観点

目標 loss を 2.0 bits/char 以下に抑えるなら：
 - 10k tokens → 過学習傾向
 - 50k tokens → 安定
 - 100k tokens → 十分

⸻

6. 結果まとめ

目的	必要トークン数
最小学習	1万
安定収束	5万
良好汎化	10万以上

16文字系列換算：

目的	サンプル数
最小	約600
安定	約3000
良好	約6000以上


⸻

最終結論

現実的には 5万〜10万文字（約3,000〜6,000個の16文字コード例）あれば、この規模のモデルで十分学習可能。

これ以下では過学習または不安定になりやすい。

