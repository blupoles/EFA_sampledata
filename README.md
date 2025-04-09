# EFA Sample Data Generator (5 Factors / 20 Questions): Car Brand Image Survey
# (探索的因子分析 練習用サンプルデータ生成スクリプト [5因子 / 20問]: 自動車ブランドイメージ調査)

## 概要 (Overview)

このPythonスクリプトは、探索的因子分析 (EFA: Exploratory Factor Analysis) の学習・練習用に特化したサンプルデータを生成します。
データは、消費者の自動車メーカーに対するブランドイメージ調査を想定しており、**5つの潜在因子**と**20の質問項目**（各因子4項目）で構成されています。

探索的分析の練習に適するように、意図的に因子構造を少し曖昧にしています（交差負荷などを含む）。CSVファイルのヘッダーには、質問内容を要約した**短いキーワード**を使用しており、分析ソフトウェアでの扱いやすさを向上させています。

## データの特徴 (Data Features)

* **サンプルサイズ:** 2000件
* **質問項目数:** 20問 (各因子4問)
* **回答尺度:** 5段階評価 (1: まったくそう思わない 〜 5: 非常にそう思う)
* **想定潜在因子数 (生成時):** 5因子（下記参照。分析時には未知として扱います）
    1.  信頼性・品質
    2.  革新性・技術力
    3.  デザイン・高級感
    4.  安全性
    5.  環境性能
* **出力形式:** CSV (Comma-Separated Values) - 短縮ヘッダー付き

## 要件 (Requirements)

* Python 3.x
* NumPy ライブラリ (`pip install numpy`)
* Pandas ライブラリ (`pip install pandas`)

## 使い方 (Usage)

1.  **スクリプトの保存:** このリポジトリにあるPythonスクリプト（例: `generate_car_data_5f_20q.py`）をローカルマシンに保存します。
2.  **仮想環境の準備 (推奨):** プロジェクト用に仮想環境を作成し、有効化します。
    ```bash
    cd path/to/your/script/directory
    python3 -m venv venv
    source venv/bin/activate # (macOS/Linux)
    # venv\Scripts\activate # (Windows)
    ```
3.  **依存ライブラリのインストール:**
    ```bash
    pip install numpy pandas
    ```
4.  **スクリプトの実行:**
    ```bash
    python generate_car_data_5f_20q.py
    ```
    スクリプト実行時に、ターミナルにもヘッダーと質問文の対応リスト（抜粋）が表示されます。

## 出力 (Output)

* スクリプトを実行したディレクトリに `car_brand_efa_data_5f_20q.csv` という名前のCSVファイルが生成されます。
* ファイルの内容:
    * 1列目: `Respondent_ID` (回答者ID: 1から2000)
    * 2列目以降: 各質問への回答 (1から5の整数値)。列ヘッダーは質問内容を表す短いキーワードになっています（下記参照）。

## ヘッダーと質問文の対応 (Header and Question Mapping)

CSVファイルのヘッダーに使用されている短いキーワードと、元の質問文の対応は以下の通りです。分析の際に参照してください。

* `故障・耐久性重視`: Q1: 自動車選びでは、故障の少なさや耐久性を最も重視する
* `品質＝信頼性`: Q2: 品質の高さが車の信頼性につながると思う
* `作りの質`: Q3: 車の作り込みの丁寧さや部品の質に関心がある
* `サービス・保証`: Q4: アフターサービスや保証の手厚さも品質の一部だと思う
* `運転支援技術`: Q5: 最新の運転支援技術に関心がある
* `コネクト機能`: Q6: コネクテッド機能やインフォテイメントを重視する
* `新技術採用評価`: Q7: 新しい技術を積極的に採用するメーカーを評価する
* `技術先進性イメージ`: Q8: そのメーカーが技術的に先進的だと感じる
* `外観デザイン重視`: Q9: 外観のデザインやスタイルは非常に重要だ
* `ステータス`: Q10: ステータスを感じさせるブランドに乗りたい
* `内装品質デザイン`: Q11: 内装のデザイン性や質感の高さを重視する
* `デザイン優越感`: Q12: 乗っていて優越感を感じられるデザインだと思う
* `安全性最重視`: Q13: 車選びで最も重要なのは安全性だと思う
* `衝突・予防安全`: Q14: 衝突安全性能や予防安全技術を重視する
* `家族のための安全`: Q15: 家族を乗せるので安全性の高い車を選びたい
* `第三者安全評価`: Q16: 第三者機関による安全評価の高さを気にする
* `燃費・排ガス`: Q17: 燃費の良さや排気ガスのクリーンさを重視する
* `EV/HV関心`: Q18: 電気自動車(EV)やハイブリッド車(HV)に関心がある
* `メーカー環境取組`: Q19: 環境問題への取り組みが進んでいるメーカーを評価する
* `生産プロセス環境`: Q20: 生産過程も含めた環境負荷低減に関心がある

## 注意事項・免責事項 (Notes/Disclaimer)

* このデータは、**シミュレーションによって生成された架空のデータ**であり、実際の調査データではありません。因子分析の学習・練習目的での利用を想定しています。
* データ生成時には内部的に5因子構造（交差負荷を含む）を設定していますが、探索的因子分析の練習としては、**この内部構造は未知であるという前提**で、データから構造を見出すプロセスを試みてください。（分析の結果、5因子が抽出されるとは限りません）
* 生成されたデータは、各種統計ソフトウェア（SPSS, R, SASなど）やPythonのライブラリ（`factor_analyzer`, `scikit-learn`など）で因子分析を行うための入力として利用できます。
