import os
import pandas as pd
from sklearn.metrics import roc_curve, auc
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def calculate_precision(df, n):
    if n == 0:
        return 0.0
    top_n = df.head(n)
    relevant_count = top_n['indication'].sum()
    return relevant_count / n


def calculate_recall(df, n):
    if n == 0:
        return 0.0
    top_n = df.head(n)
    relevant_count = top_n['indication'].sum()
    total_relevant = df['indication'].sum()
    return relevant_count / total_relevant if total_relevant > 0 else 0.0


results = []

lambda_list = [0.1,0.3,0.4,0.5,0.7,1.0,1.2,1.7,5]

for lam in lambda_list:
    df = pd.read_csv(
        #fr'output/proximity of different λ/Chemical-Disease/CTD_λ_{lam}_with_indication.csv'
        #fr'output/proximity of different λ/drug_disease/drug_λ_{lam}_with_indication.csv'
        fr'output/proximity of different λ/Herb-Symptom/TCM_λ_{lam}_with_indication.csv'
    )

    # d 越小越相关
    df = df.sort_values(by='d', ascending=True)

    # ROC
    fpr_d, tpr_d, _ = roc_curve(df['indication'], -df['d'])
    fpr_z, tpr_z, _ = roc_curve(df['indication'], -df['z'])

    roc_auc_d = auc(fpr_d, tpr_d)
    roc_auc_z = auc(fpr_z, tpr_z)

    # Top 1%
    num = int(len(df) * 0.01)

    precision = calculate_precision(df, num)
    recall = calculate_recall(df, num)

    results.append([
        lam,
        roc_auc_d,
        roc_auc_z,
        precision,
        recall
    ])

    print(f"λ = {lam}")
    print(f"AUC_d = {roc_auc_d:.4f}")
    print(f"AUC_z = {roc_auc_z:.4f}")
    print(f"Precision@Top1% = {precision:.5f}")
    print(f"Recall@Top1% = {recall:.5f}")
    print('-' * 40)


# 构建结果 DataFrame（循环结束后）
results_df = pd.DataFrame(
    results,
    columns=['λ', 'AUC_d', 'AUC_z', 'Precision@Top1%_d', 'Recall@Top1%_d']
)

# 保留小数
results_df['AUC_d'] = results_df['AUC_d'].round(4)
results_df['AUC_z'] = results_df['AUC_z'].round(4)
results_df['Precision@Top1%_d'] = results_df['Precision@Top1%_d'].round(5)
results_df['Recall@Top1%_d'] = results_df['Recall@Top1%_d'].round(5)

# 保存到 Excel
new_excel_file_path = r"output/net_performance_MMI.xlsx"
#sheet_name = "CD"
#sheet_name = "DD"
sheet_name = "HS"
if os.path.exists(new_excel_file_path):
    with pd.ExcelWriter(
        new_excel_file_path,
        engine='openpyxl',
        mode='a',
        if_sheet_exists='replace'
    ) as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)
else:
    with pd.ExcelWriter(
        new_excel_file_path,
        engine='openpyxl',
        mode='w'
    ) as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)
