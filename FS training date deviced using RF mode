import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 数据读取与处理
data = pd.read_excel(r'C:\Users\Liu\Desktop\FS-YS-over.xlsx').loc[:, ~data.columns.duplicated()]
data['strain'] = data['strain'].astype(str)
data = data[~data['strain'].str.contains('over', case=False)]
data['strain'] = pd.to_numeric(data['strain'], errors='coerce').dropna()

# 特征与目标
target = "Ys"
features = [col for col in data.columns if col != target]
data = data.dropna(subset=[target] + features)
X, y = data[features], data[target]

# 训练比例与结果存储
ratios = np.arange(0.3, 0.91, 0.1)
res = {r: [] for r in ratios}

# 模型定义
def rf():
    return RandomForestRegressor(100, max_depth=5, random_state=42, n_jobs=-1)

# 100次实验
for r in tqdm(ratios, desc="进度"):
    for i in range(100):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=r, random_state=42+i)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        m = rf()
        m.fit(X_tr_s, y_tr)
        res[r].append(np.sqrt(mean_squared_error(y_te, m.predict(X_te_s))))

# 结果汇总
summary = []
for r in ratios:
    rmse = res[r]
    summary.append({
        '训练比例': f"{r:.0%}",
        'RMSE均值': np.mean(rmse),
        'RMSE标准差': np.std(rmse),
        '次数': len(rmse)
    })

df = pd.DataFrame(summary).sort_values('训练比例')

# 输出
print("\nRF模型性能（目标：Ys，100次）：")
print(f"样本数: {len(data)}, 特征数: {len(features)}")
print("已去含'over'数据")
print("-" * 70)
print(f"{'训练比例':<10} | {'RMSE均值':<10} | {'RMSE标准差':<15} | {'次数':<5}")
print("-" * 70)
for _, row in df.iterrows():
    print(f"{row['训练比例']:<10} | {row['RMSE均值']:.4f}      | {row['RMSE标准差']:.4f}            | {row['次数']:<5}")

# 保存
df.to_excel(r'C:\Users\Liu\Desktop\RF-结果(无over).xlsx', index=False)
print(f"\n结果存至: {r'C:\Users\Liu\Desktop\RF-结果(无over).xlsx'}")
