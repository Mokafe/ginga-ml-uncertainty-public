# ginga-ml-uncertainty（自由にどうぞ）

『銀河鉄道の夜』の雰囲気を借りて、回帰とかカウントとか不確実性とかを **手を動かして確認する** 個人研究ノートです。  
読書会のネタ・授業の小道具・実験・改変、ぜんぶ自由にどうぞ（クレジットを残してくれると嬉しいです）。

> 合言葉：**深呼吸してから検証する。**  
> まず観測条件（＝不確実性）を疑う。

---

## これ、何ができるの？
ざっくり言うと、銀河鉄道の窓からの「観測」をデータにして、

- **ぼんやり白さ（haze_obs）**を回帰で説明してみる
- **星の数（star_count）**を「カウント」として扱って、線形回帰とポアソン回帰の違いを体感する
- overlap（レンズの重なり）で「予測の幅」が変わるのを見て、**不確実性って何？**を掴む
- 交互作用・多重共線性・変数選択・外れ値検知まで、軽く触る
- おまけで PyTorch（Poisson NLL）にもつなぐ

…という感じです。

---

## いちばん早い入口（Colab）
リポジトリを作ったら、READMEにこのバッジを置くとワンクリックで開けます（`<USER>/<REPO>` は差し替え）：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mokafe/ginga-ml-uncertainty-public/blob/main/notebooks/01_galaxy_haze_public_colab.ipynb)

↑ ここからそのまま開けます。


---

## だいたいの世界観（データ）
1行 = 銀河鉄道の窓からの観測メモ（本文から拾った文）です。

- `u`：銀河帯の位置（中心に近いほど“厚い”想定）
- `overlap`：レンズの重なり（鮮明さ）
- `star_count`：見える星の数（非負の整数）
- `haze_obs`：ぼんやり白さ（連続値）
- `memo`：観測メモ（本文由来）

“真値”っぽい列（`thickness_true` など）は、チェック用のおまけです。  
隠したまま遊ぶのも、見ながら納得するのも、どっちでもOK。

---

## 置いてあるもの
- `notebooks/01_galaxy_haze_public_colab.ipynb`  
  Colabでそのまま動く版（plotlyで可視化も出ます）

- `notebooks/01_galaxy_haze_handson.ipynb`  
  ローカルのJupyter向け。落ち着いて読みながら進める版。

- `notebooks/01_galaxy_haze_with_ground_truth.ipynb`  
  “真値”列も含めて、生成過程のチェックをしながら進める版。

- `src/make_dataset.py`  
  テキストを入力に、v2のCSV/JSONLを生成します（seed固定）。

- `data/sample/`  
  動作確認用の少量データ（数十〜百行）。

---

## 使い方（最短）
### 1) 環境
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) データ生成
作品本文は同梱しません。各自で用意して `data/raw/ginga.txt` に置く想定です。

```bash
python src/make_dataset.py \
  --text data/raw/ginga.txt \
  --out_csv data/ginga_galaxy_haze_v2.csv \
  --out_jsonl data/ginga_galaxy_haze_v2.jsonl \
  --seed 10 \
  --add_event
```

### 3) ノートを開く
```bash
jupyter lab
```
`notebooks/` を開いて、好きなやつからどうぞ。

---

## ここで掴みたい論点（メモ）
- 線形回帰 vs ポアソン回帰、どっちが “銀河っぽい”説明になる？
- overlap が小さいとき、予測の“幅”（不確実性）は増える？
- 交互作用を入れると何が見える？ その代わり何が難しくなる？
- 外れ値は消す？ 読む？（ここが楽しい）
- もし星図が画像なら：CNN＋どんな損失にする？

---

## 注意（ゆるく）
- 作品本文やPDF（ISLR/D2Lなど）はここに同梱しません。各自で入手してください。
- これは個人研究メモ寄りなので、厳密さより「試して分かる」を優先しています。

---

## ライセンス
コードはMIT（`LICENSE`）。サンプルデータはこのリポジトリのための生成物です。  
作品本文は各自で用意してください。
