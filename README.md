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

## 俳句モデルの作成

青空文庫から俳句を抽出したaozora_haiku_30000.txt(約4000句)を学習させた.  
microgpt.pyは26文字の英字のみを想定しているため、日本語を学習させるには最低限パラメータ調整しないとまともな俳句が出にくい.

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

