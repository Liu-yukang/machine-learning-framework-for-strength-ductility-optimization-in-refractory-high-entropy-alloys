import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                             AdaBoostRegressor, VotingRegressor, StackingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

data = pd.read_excel(r'C:\Users\Liu\Desktop\FS-OVER.xlsx')
data = data.loc[:, ~data.columns.duplicated()]
data['strain'] = data['strain'].astype(str)
data = data[~data['strain'].str.contains('over', case=False, na=False)]
data['strain'] = pd.to_numeric(data['strain'], errors='coerce')
data = data.dropna(subset=['strain'])

X = data.drop('strain', axis=1)
y = data['strain']

base_models = {
    'RF': RandomForestRegressor(100, random_state=42),
    'KNN': KNeighborsRegressor(),
    'NN': MLPRegressor(random_state=42, max_iter=1000),
    'LR': LinearRegression(),
    'XGBT': XGBRegressor(random_state=42),
    'SVC.ploy': SVR(kernel='poly'),
    'SVC.rbf': SVR(kernel='rbf'),
    'SVR.l': SVR(kernel='linear'),
    'MLP': MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50)),
    'LGBM': LGBMRegressor(random_state=42),
    'GBDT': GradientBoostingRegressor(random_state=42),
    'Adaboost': AdaBoostRegressor(random_state=42)
}

voting_model = VotingRegressor([('RF', base_models['RF']),
                               ('XGBT', base_models['XGBT']),
                               ('LGBM', base_models['LGBM']),
                               ('GBDT', base_models['GBDT'])])

stacking_model = StackingRegressor([('RF', base_models['RF']),
                                   ('XGBT', base_models['XGBT']),
                                   ('LGBM', base_models['LGBM']),
                                   ('SVC.rbf', base_models['SVC.rbf'])],
                                  final_estimator=LinearRegression())

models = {**base_models, 'Voting': voting_model, 'Stacking': stacking_model}
results = {name: {'RMSE': [], 'R2': []} for name in models}

num_iterations = 100
for _ in tqdm(range(num_iterations), desc="进度"):
    kf = KFold(5, shuffle=True, random_state=np.random.randint(0, 10000))
    for name, model in models.items():
        rmse, r2 = [], []
        for tr_idx, te_idx in kf.split(X):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            model.fit(X_tr, y_tr)
            y_pr = model.predict(X_te)
            rmse.append(np.sqrt(mean_squared_error(y_te, y_pr)))
            r2.append(r2_score(y_te, y_pr))
        results[name]['RMSE'].append(np.mean(rmse))
        results[name]['R2'].append(np.mean(r2))

print("\n模型性能（100次五折交叉验证）：")
print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
print("-" * 80)
print(f"{'模型':<12} | {'RMSE均值':<10} | {'RMSE标准差':<12} | {'R2均值':<10} | {'R2标准差':<12}")
print("-" * 80)
for name, met in results.items():
    print(f"{name:<12} | {np.mean(met['RMSE']):.4f}      | {np.std(met['RMSE']):.4f}        | {np.mean(met['R2']):.4f}      | {np.std(met['R2']):.4f}")
