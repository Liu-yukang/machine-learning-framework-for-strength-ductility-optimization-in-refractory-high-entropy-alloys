import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tqdm import tqdm

features = [
    'number', 'VEC', 'Tm', 'ρ', 'Ec', 'w6', 'r', 'δr', 'D.r',
    'χ', 'δχ', 'D.χ', 'G', 'δG', 'D.G', 'B', 'δB', 'D.B',
    'v', 'δv', 'D.v', 'E', 'Smix', 'Hmix', 'Gmix', 'Ω', 'Λ', 'γ',
    'ƞ', 'μ', 'A', 'F', 'a', 'Δa'
]

file_path = r'C:\Users\Liu\Desktop\666666.xlsx'
data = pd.read_excel(file_path).loc[:, ~data.columns.duplicated()]
target = "Ys"

results = []

for feature in tqdm(features, desc="进度"):
    X = data[[feature]]
    y = data[target]
    
    model = RandomForestRegressor(100, max_depth=5, random_state=42, n_jobs=-1)
    
    r2_scores = []
    kf = KFold(10, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        mean_tr = X_tr.mean().item()
        std_tr = X_tr.std().item()
        
        if std_tr == 0:
            X_tr_s = X_tr - mean_tr
            X_te_s = X_te - mean_tr
        else:
            X_tr_s = (X_tr - mean_tr) / std_tr
            X_te_s = (X_te - mean_tr) / std_tr
        
        model.fit(X_tr_s, y_tr)
        y_pr = model.predict(X_te_s)
        r2_scores.append(r2_score(y_te, y_pr))
    
    mean_r2 = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    results.append({'特征': feature, '平均R²': mean_r2, 'R²误差': r2_std})

print("各特征单独建模平均R²及误差（原始顺序）：")
for res in results:
    print(f"{res['特征']}: 平均R² = {res['平均R²']:.4f}, 误差 = {res['R²误差']:.4f}")

y_var = data[target].var()
print(f"\n目标{target}方差: {y_var:.4f}")

result_df = pd.DataFrame(results)
output_excel_path = r'C:\Users\Liu\Desktop\特征R²结果（按原始顺序）.xlsx'
result_df.to_excel(output_excel_path, index=False)
print(f"\n结果存至: {output_excel_path}")
