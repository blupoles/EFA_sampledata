import numpy as np
import pandas as pd

# --- 設定 ---
n_samples = 2000  # サンプル数
n_questions = 20 # 質問数 (5因子 x 4問)
n_factors = 5    # 想定される潜在因子数

# --- 各質問文 (参考用) ---
question_texts_full = [
    # F1: 信頼性・品質 (Q1-Q4)
    "Q1: 自動車選びでは、故障の少なさや耐久性を最も重視する",
    "Q2: 品質の高さが車の信頼性につながると思う",
    "Q3: 車の作り込みの丁寧さや部品の質に関心がある",
    "Q4: アフターサービスや保証の手厚さも品質の一部だと思う",
    # F2: 革新性・技術力 (Q5-Q8)
    "Q5: 最新の運転支援技術に関心がある", # 旧Q4
    "Q6: コネクテッド機能やインフォテイメントを重視する", # 旧Q5
    "Q7: 新しい技術を積極的に採用するメーカーを評価する", # 旧Q6
    "Q8: そのメーカーが技術的に先進的だと感じる",
    # F3: デザイン・高級感 (Q9-Q12)
    "Q9: 外観のデザインやスタイルは非常に重要だ", # 旧Q7
    "Q10: ステータスを感じさせるブランドに乗りたい", # 旧Q8
    "Q11: 内装のデザイン性や質感の高さを重視する", # 旧Q9
    "Q12: 乗っていて優越感を感じられるデザインだと思う",
    # F4: 安全性 (Q13-Q16)
    "Q13: 車選びで最も重要なのは安全性だと思う", # 旧Q10
    "Q14: 衝突安全性能や予防安全技術を重視する", # 旧Q11
    "Q15: 家族を乗せるので安全性の高い車を選びたい", # 旧Q12
    "Q16: 第三者機関による安全評価の高さを気にする",
    # F5: 環境性能 (Q17-Q20)
    "Q17: 燃費の良さや排気ガスのクリーンさを重視する", # 旧Q13
    "Q18: 電気自動車(EV)やハイブリッド車(HV)に関心がある", # 旧Q14
    "Q19: 環境問題への取り組みが進んでいるメーカーを評価する", # 旧Q15
    "Q20: 生産過程も含めた環境負荷低減に関心がある"
]

# --- 短いヘッダーラベル (20問分) ---
short_headers = [
    "故障・耐久性重視",    # Q1 (F1)
    "品質＝信頼性",        # Q2 (F1)
    "作りの質",            # Q3 (F1)
    "サービス・保証",      # Q4 (F1) - NEW
    "運転支援技術",        # Q5 (F2)
    "コネクト機能",        # Q6 (F2)
    "新技術採用評価",      # Q7 (F2)
    "技術先進性イメージ",  # Q8 (F2) - NEW
    "外観デザイン重視",    # Q9 (F3)
    "ステータス",          # Q10 (F3)
    "内装品質デザイン",    # Q11 (F3)
    "デザイン優越感",      # Q12 (F3) - NEW
    "安全性最重視",        # Q13 (F4)
    "衝突・予防安全",      # Q14 (F4)
    "家族のための安全",    # Q15 (F4)
    "第三者安全評価",      # Q16 (F4) - NEW
    "燃費・排ガス",        # Q17 (F5)
    "EV/HV関心",         # Q18 (F5)
    "メーカー環境取組",    # Q19 (F5)
    "生産プロセス環境"     # Q20 (F5) - NEW
]


# --- 潜在変数の生成 ---
np.random.seed(42) # 再現性
factor_scores = np.random.normal(loc=0, scale=1, size=(n_samples, n_factors))

# --- 因子負荷行列の調整 (20x5, EFA向け) ---
# 交差負荷を含め、負荷の強弱をつける
loadings_matrix = np.zeros((n_questions, n_factors))

main_loading_high = 0.65 # 少し下げて交差負荷の余地を作る
main_loading_mid = 0.55
cross_loading_mod = 0.25
cross_loading_low = 0.15

# F1 Items (Q1-4)
loadings_matrix[0, 0] = main_loading_high
loadings_matrix[1, 0] = main_loading_mid
loadings_matrix[2, 0] = main_loading_mid
loadings_matrix[2, 2] = cross_loading_low  # Q3(作りの質) -> F3(デザイン)
loadings_matrix[3, 0] = main_loading_high  # Q4(サービス) -> 主にF1

# F2 Items (Q5-8)
loadings_matrix[4, 1] = main_loading_mid   # Q5(運転支援)
loadings_matrix[5, 1] = main_loading_mid   # Q6(コネクト)
loadings_matrix[6, 1] = main_loading_high  # Q7(新技術採用)
loadings_matrix[7, 1] = main_loading_mid   # Q8(技術先進イメージ)
loadings_matrix[7, 2] = cross_loading_mod # Q8 -> F3(先進イメージ->ステータス)

# F3 Items (Q9-12)
loadings_matrix[8, 2] = main_loading_high  # Q9(外観)
loadings_matrix[9, 2] = main_loading_mid   # Q10(ステータス)
loadings_matrix[10, 2] = main_loading_mid  # Q11(内装)
loadings_matrix[10, 0] = cross_loading_mod # Q11 -> F1(内装品質->品質)
loadings_matrix[11, 2] = main_loading_high # Q12(優越感) -> 主にF3

# F4 Items (Q13-16)
loadings_matrix[12, 3] = main_loading_high # Q13(安全性最重視)
loadings_matrix[13, 3] = main_loading_mid  # Q14(衝突予防)
loadings_matrix[14, 3] = main_loading_mid  # Q15(家族安全)
loadings_matrix[14, 0] = cross_loading_low # Q15 -> F1(家族安全->信頼性)
loadings_matrix[15, 3] = main_loading_high # Q16(第三者評価) -> 主にF4

# F5 Items (Q17-20)
loadings_matrix[16, 4] = main_loading_high # Q17(燃費排ガス)
loadings_matrix[17, 4] = main_loading_mid  # Q18(EV/HV)
loadings_matrix[17, 1] = cross_loading_mod # Q18 -> F2(EV/HV->技術)
loadings_matrix[18, 4] = main_loading_mid  # Q19(メーカー取組)
loadings_matrix[19, 4] = main_loading_high # Q20(生産プロセス) -> 主にF5

# 誤差（各質問固有のばらつき）
error_std = 0.58 # 項目数が増えたので少し誤差を増やす

# --- 回答データの生成 ---
base_data = np.dot(factor_scores, loadings_matrix.T) + \
            np.random.normal(loc=0, scale=error_std, size=(n_samples, n_questions))
scale_mean = 3.0
scale_std_factor = 1.0
scaled_data = base_data * scale_std_factor + scale_mean
clipped_data = np.clip(scaled_data, 1, 5)
response_data = np.round(clipped_data).astype(int)

# --- DataFrameの作成 ---
df = pd.DataFrame(response_data, columns=short_headers)
df.insert(0, 'Respondent_ID', range(1, n_samples + 1))

# --- CSVファイルとして保存 ---
output_filename = 'car_brand_efa_data_5f_20q.csv' # 5因子20問用ファイル名
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print(f"探索的因子分析(EFA)向けのサンプルデータ(5因子20問版)が '{output_filename}' として保存されました。")
print("CSVのヘッダー行には、質問内容を表す短いラベルが含まれています。")
print(f"データ形状: {df.shape}")
print("--- ヘッダーラベルと質問文の対応 (抜粋表示) ---")
for i in range(5): # 最初の5問だけ表示（長くなるため）
    print(f"{short_headers[i]}: {question_texts_full[i]}")
print("... (対応の完全なリストはREADMEを確認してください) ...")
print("------------------------------------")
print("最初の5行:")
print(df.head())
