import numpy as np
import pandas as pd

# --- 設定 ---
n_samples = 2000
n_questions = 20 # 5因子 x 4問
n_factors = 5

# --- データタイプ選択 ---
# 'clear', 'ambiguous', または 'subtle' を設定
# 'clear':     構造が明確。基本学習向き。
# 'ambiguous': やや曖昧。一般的なEFA練習向き。
# 'subtle':    さらに曖昧。因子数の決定がより難しい、挑戦的な練習向き。
data_clarity = 'subtle'  # <<<--- ここを 'clear', 'ambiguous', 'subtle' に変更

# --- パラメータ設定 (データタイプに応じて変更) ---
if data_clarity == 'clear':
    main_loading_high = 0.80
    main_loading_mid = 0.75
    cross_loading = 0.15
    error_std = 0.40
    output_suffix = 'clear_structure'
    print("データタイプ: 明確な構造 (基本学習向き)")
elif data_clarity == 'ambiguous':
    main_loading_high = 0.65
    main_loading_mid = 0.55
    cross_loading = 0.25
    error_std = 0.58
    output_suffix = 'ambiguous_structure'
    print("データタイプ: やや曖昧な構造 (EFA実践練習向き)")
elif data_clarity == 'subtle':
    # 'ambiguous' よりさらに構造を弱める
    main_loading_high = 0.60 # やや下げる
    main_loading_mid = 0.50  # やや下げる
    cross_loading = 0.28     # 交差負荷は維持または少し上げる
    error_std = 0.65         # 誤差を上げる
    output_suffix = 'subtle_structure'
    print("データタイプ: 微妙な構造 (因子数決定の挑戦的練習向き)")
else:
    raise ValueError("data_clarity は 'clear', 'ambiguous', 'subtle' のいずれかを設定してください。")

# --- 各質問文 (参考用) ---
question_texts_full = [
    "Q1:...", "Q2:...", "Q3:...", "Q4:...", "Q5:...", "Q6:...", "Q7:...", "Q8:...", "Q9:...", "Q10:...",
    "Q11:...", "Q12:...", "Q13:...", "Q14:...", "Q15:...", "Q16:...", "Q17:...", "Q18:...", "Q19:...", "Q20:..."
    # 省略: 完全なリストは以前の回答を参照
]
# (完全なリストをここに入れておく)
question_texts_full = [
    "Q1: 自動車選びでは、故障の少なさや耐久性を最も重視する", "Q2: 品質の高さが車の信頼性につながると思う", "Q3: 車の作り込みの丁寧さや部品の質に関心がある", "Q4: アフターサービスや保証の手厚さも品質の一部だと思う",
    "Q5: 最新の運転支援技術に関心がある", "Q6: コネクテッド機能やインフォテイメントを重視する", "Q7: 新しい技術を積極的に採用するメーカーを評価する", "Q8: そのメーカーが技術的に先進的だと感じる",
    "Q9: 外観のデザインやスタイルは非常に重要だ", "Q10: ステータスを感じさせるブランドに乗りたい", "Q11: 内装のデザイン性や質感の高さを重視する", "Q12: 乗っていて優越感を感じられるデザインだと思う",
    "Q13: 車選びで最も重要なのは安全性だと思う", "Q14: 衝突安全性能や予防安全技術を重視する", "Q15: 家族を乗せるので安全性の高い車を選びたい", "Q16: 第三者機関による安全評価の高さを気にする",
    "Q17: 燃費の良さや排気ガスのクリーンさを重視する", "Q18: 電気自動車(EV)やハイブリッド車(HV)に関心がある", "Q19: 環境問題への取り組みが進んでいるメーカーを評価する", "Q20: 生産過程も含めた環境負荷低減に関心がある"
]

# --- 短いヘッダーラベル ---
short_headers = [
    "故障・耐久性重視", "品質＝信頼性", "作りの質", "サービス・保証", # F1
    "運転支援技術", "コネクト機能", "新技術採用評価", "技術先進性イメージ", # F2
    "外観デザイン重視", "ステータス", "内装品質デザイン", "デザイン優越感", # F3
    "安全性最重視", "衝突・予防安全", "家族のための安全", "第三者安全評価", # F4
    "燃費・排ガス", "EV/HV関心", "メーカー環境取組", "生産プロセス環境" # F5
]

# --- 潜在変数の生成 ---
np.random.seed(42)
factor_scores = np.random.normal(loc=0, scale=1, size=(n_samples, n_factors))

# --- 因子負荷行列の構築 (データタイプに応じて調整) ---
loadings_matrix = np.zeros((n_questions, n_factors))

# 主負荷の設定
# F1 Items (Q1-4)
loadings_matrix[0, 0] = main_loading_high
loadings_matrix[1, 0] = main_loading_mid
loadings_matrix[2, 0] = main_loading_mid
loadings_matrix[3, 0] = main_loading_high
# F2 Items (Q5-8)
loadings_matrix[4, 1] = main_loading_mid
loadings_matrix[5, 1] = main_loading_mid
loadings_matrix[6, 1] = main_loading_high
loadings_matrix[7, 1] = main_loading_mid
# F3 Items (Q9-12)
loadings_matrix[8, 2] = main_loading_high
loadings_matrix[9, 2] = main_loading_mid
loadings_matrix[10, 2] = main_loading_mid
loadings_matrix[11, 2] = main_loading_high
# F4 Items (Q13-16)
loadings_matrix[12, 3] = main_loading_high
loadings_matrix[13, 3] = main_loading_mid
loadings_matrix[14, 3] = main_loading_mid
loadings_matrix[15, 3] = main_loading_high
# F5 Items (Q17-20)
loadings_matrix[16, 4] = main_loading_high
loadings_matrix[17, 4] = main_loading_mid
loadings_matrix[18, 4] = main_loading_mid
loadings_matrix[19, 4] = main_loading_high

# 交差負荷の設定 (データタイプに応じて調整)
if data_clarity == 'clear':
    # 交差負荷は少なく、弱く
    loadings_matrix[7, 2] = cross_loading # Q8 -> F3
    loadings_matrix[10, 0] = cross_loading # Q11 -> F1
    loadings_matrix[17, 1] = cross_loading # Q18 -> F2
elif data_clarity == 'ambiguous' or data_clarity == 'subtle':
    # 'ambiguous' と 'subtle' で同じ位置に交差負荷を設定
    # 値 (cross_loading) は各モードで異なる値が使われる
    cross_loading_low = cross_loading * 0.6 # 弱い交差負荷用 (主に'subtle'で影響)
    loadings_matrix[2, 2] = cross_loading_low # Q3(作り) -> F3(デザイン)
    loadings_matrix[7, 2] = cross_loading     # Q8(技術先進イメージ) -> F3(デザイン/高級感)
    loadings_matrix[10, 0] = cross_loading    # Q11(内装) -> F1(品質)
    loadings_matrix[14, 0] = cross_loading_low # Q15(家族安全) -> F1(信頼性)
    loadings_matrix[17, 1] = cross_loading    # Q18(EV/HV) -> F2(技術)
    # 'subtle' の場合、さらに曖昧さを増すために、もう一つ弱い交差負荷を追加してもよい
    if data_clarity == 'subtle':
         loadings_matrix[4, 3] = cross_loading_low # Q5(運転支援) -> F4(安全)？ ほんの少しだけ

# --- 回答データの生成 ---
base_data = np.dot(factor_scores, loadings_matrix.T) + \
            np.random.normal(loc=0, scale=error_std, size=(n_samples, n_questions))
scale_mean = 3.0
scale_std_factor = 1.0
scaled_data = base_data * scale_std_factor + scale_mean
clipped_data = np.clip(scaled_data, 1, 5)
response_data = np.round(clipped_data).astype(int)

# --- DataFrameの作成 & 保存 ---
df = pd.DataFrame(response_data, columns=short_headers)
df.insert(0, 'Respondent_ID', range(1, n_samples + 1))
output_filename = f'car_brand_efa_data_{output_suffix}.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"サンプルデータが '{output_filename}' として保存されました。")
print(f"データ形状: {df.shape}")
# (対応表や head の表示は前回同様)
print("--- ヘッダーラベルと質問文の対応 (抜粋表示) ---")
for i in range(5):
    print(f"{short_headers[i]}: {question_texts_full[i]}")
print("... (対応の完全なリストはREADMEを確認してください) ...")
print("------------------------------------")
print("最初の5行:")
print(df.head())
