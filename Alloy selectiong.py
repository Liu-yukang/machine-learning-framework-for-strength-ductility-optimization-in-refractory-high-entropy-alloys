import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from tqdm import tqdm
import time
import random
import scipy.stats as stats
import os
import joblib
from itertools import combinations

warnings.filterwarnings("ignore")

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
intermediate_path = os.path.join(desktop_path, "RHEA_Optimization_Intermediate")
os.makedirs(intermediate_path, exist_ok=True)

strength_train_data_path = r'C:\Users\Liu\Desktop\FS-YS-over.xlsx'
strength_features = ['D.r', 'Hmix', 'Λ', 'ƞ', 'a']
strength_target = 'Ys'

strain_train_data_path = r'C:\Users\Liu\Desktop\FS-YS-over.xlsx'
strain_features = ['γ', 'δχ', 'r', 'Hmix', 'ρ']
strain_target = 'strain'

strength_df = pd.read_excel(strength_train_data_path)
strength_df = strength_df.dropna(subset=strength_features + [strength_target])
strength_df = strength_df[~strength_df[strength_target].astype(str).str.contains('over', case=False, na=False)]
X_strength_train = strength_df[strength_features]
y_strength_train = strength_df[strength_target]

strain_df = pd.read_excel(strain_train_data_path)
strain_df = strain_df.dropna(subset=strain_features + [strain_target])
strain_df = strain_df[~strain_df[strain_target].astype(str).str.contains('over', case=False, na=False)]
X_strain_train = strain_df[strain_features]
y_strain_train = strain_df[strain_target]

strength_scaler = StandardScaler()
X_strength_scaled = strength_scaler.fit_transform(X_strength_train)

strain_scaler = StandardScaler()
X_strain_scaled = strain_scaler.fit_transform(X_strain_train)

current_best_strength = y_strength_train.max()
current_best_strain = y_strain_train.max()

RF_HYPERPARAMS_STRENGTH = {
    "n_estimators": 213,
    "max_depth": 15,
    "min_samples_split": 1,
    "min_samples_leaf": 2,
    "max_features": 'log2',
    "bootstrap": True
}

RF_HYPERPARAMS_STRAIN = {
    "n_estimators": 117,
    "max_depth": 14,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 'log2'
}

element_properties = {
    'Al': {'r': 1.432, 'χ': 1.61, 'B': 76, 'G': 26, 'v': 0.35, 'density': 2.70, 'a': 4.050, 'VEC': 3, 'Tm': 933},
    'Hf': {'r': 1.578, 'χ': 1.3, 'B': 112, 'G': 28, 'v': 0.38, 'density': 13.31, 'a': 3.559, 'VEC': 4, 'Tm': 2506},
    'Mo': {'r': 1.363, 'χ': 2.16, 'B': 232.5, 'G': 123, 'v': 0.31, 'density': 10.28, 'a': 3.147, 'VEC': 6, 'Tm': 2896},
    'Nb': {'r': 1.429, 'χ': 1.60, 'B': 172.5, 'G': 38, 'v': 0.39, 'density': 8.57, 'a': 3.301, 'VEC': 5, 'Tm': 2750},
    'Ta': {'r': 1.43, 'χ': 1.5, 'B': 202.5, 'G': 69, 'v': 0.345, 'density': 16.68, 'a': 3.303, 'VEC': 5, 'Tm': 3290},
    'Ti': {'r': 1.462, 'χ': 1.54, 'B': 112.5, 'G': 44, 'v': 0.33, 'density': 4.51, 'a': 3.276, 'VEC': 4, 'Tm': 1941},
    'V': {'r': 1.343, 'χ': 1.63, 'B': 160, 'G': 47, 'v': 0.37, 'density': 6.11, 'a': 3.039, 'VEC': 5, 'Tm': 2183},
    'Zr': {'r': 1.603, 'χ': 1.33, 'B': 96, 'G': 33, 'v': 0.35, 'density': 6.51, 'a': 3.582, 'VEC': 4, 'Tm': 2128},
    'W': {'r': 1.367, 'χ': 2.36, 'B': 310, 'G': 161, 'v': 0.28, 'density': 19.25, 'a': 3.165, 'VEC': 6, 'Tm': 3695}
}

selected_elements = ['Al', 'Hf', 'Mo', 'Nb', 'Ta', 'Ti', 'V', 'Zr', 'W']

element_bounds_int = {
    'Al': (3, 25),
    'Ti': (3, 38),
    'Hf': (8, 35),
    'Mo': (8, 35),
    'Nb': (1, 40),
    'Ta': (3, 37),
    'V': (3, 35),
    'Zr': (2, 35),
    'W': (3, 35)
}

mix_enthalpy_matrix = {
    'Al': [None, -39, -5, -18, -19, -30, -16, -44, -2],
    'Hf': [-39, None, -4, 4, 3, 0, -7, 0, -4],
    'Mo': [-5, -4, None, -6, -5, -4, -2, -6, 0],
    'Nb': [-18, 4, -6, None, 0, 2, -1, 4, -6],
    'Ta': [-19, 3, -5, 0, None, 1, -2, 3, -5],
    'Ti': [-30, 0, -4, 2, 1, None, -2, 0, -4],
    'V': [-16, -7, -2, -1, -2, -2, None, -7, -1],
    'Zr': [-44, 0, -6, 4, 3, 0, -7, None, -6],
    'W': [-2, -4, 0, -6, -5, -4, -1, -6, None]
}
mix_enthalpy_df = pd.DataFrame(mix_enthalpy_matrix, index=selected_elements)

def latin_hypercube_sampling(n_samples, element_list):
    samples = []
    for _ in range(n_samples):
        num_elements = random.randint(4, 6)
        selected_elements = random.sample(element_list, num_elements)
        
        composition = {}
        for element in element_list:
            composition[element] = 0.0
            
        remaining = 100.0
        bounds = [element_bounds_int[e] for e in selected_elements]
        
        for i, element in enumerate(selected_elements[:-1]):
            min_val, max_val = bounds[i]
            available_max = min(max_val, remaining - sum(bounds[j][0] for j in range(i+1, len(selected_elements))))
            available_min = max(min_val, remaining - sum(bounds[j][1] for j in range(i+1, len(selected_elements))))
            
            if available_min > available_max:
                value = (available_min + available_max) / 2
            else:
                value = random.uniform(available_min, available_max)
            
            composition[element] = value
            remaining -= value
        
        composition[selected_elements[-1]] = remaining
        
        samples.append(composition)
    
    return samples

def calculate_alloy_features(alloy_comp):
    active = [e for e in alloy_comp if alloy_comp[e] > 0]
    if len(active) < 4:
        return None, None

    total = sum(alloy_comp.values())
    norm = {e: v/total for e, v in alloy_comp.items()}

    # Strength features: D.r, Hmix, Λ, η, a
    D_r = 0
    for i, e1 in enumerate(active):
        c1, r1 = norm[e1], element_properties[e1]['r']
        for j, e2 in enumerate(active):
            if i != j:
                c2, r2 = norm[e2], element_properties[e2]['r']
                D_r += c1 * c2 * abs(r1 - r2)
    
    Hmix = sum(4 * norm[e1] * norm[e2] * mix_enthalpy_df.loc[e1, e2] 
               for i, e1 in enumerate(active) for j, e2 in enumerate(active) if i < j)
    
    R = 8.314
    Smix = -R * sum(c * np.log(c) for c in norm.values() if c > 0)
    r_avg = sum(norm[e] * element_properties[e]['r'] for e in active)
    delta_r = np.sqrt(sum(norm[e] * ((1 - element_properties[e]['r']/r_avg) ** 2) for e in active))
    Lambda = Smix / (delta_r ** 2) if delta_r != 0 else 0
    
    G_avg = sum(norm[e] * element_properties[e]['G'] for e in active)
    eta_sum = 0
    for e in active:
        numerator = 2 * (element_properties[e]['G'] - G_avg) / (element_properties[e]['G'] + G_avg)
        denominator = 1 + 0.5 * abs(numerator)
        eta_sum += norm[e] * numerator / denominator
    eta = eta_sum
    
    a = sum(norm[e] * element_properties[e]['a'] for e in active)

    # Strain features: γ, δχ, r, Hmix, ρ
    radii = [element_properties[e]['r'] for e in active]
    r_min, r_max = min(radii), max(radii)
    
    numerator = 1 - np.sqrt((r_avg + r_min)**2 - r_avg**2) / (r_avg + r_min)
    denominator = 1 - np.sqrt((r_avg + r_max)**2 - r_avg**2) / (r_avg + r_max)
    gamma = numerator / denominator if denominator != 0 else 0
    
    chi_avg = sum(norm[e] * element_properties[e]['χ'] for e in active)
    d_chi = np.sqrt(sum(norm[e] * ((element_properties[e]['χ'] - chi_avg) ** 2) for e in active))
    
    rho = sum(norm[e] * element_properties[e]['density'] for e in active)

    return [D_r, Hmix, Lambda, eta, a], [gamma, d_chi, r_avg, Hmix, rho]

def calculate_ei(mu, sigma, f_max):
    if sigma == 0:
        return max(0, mu - f_max)
    z = (mu - f_max) / sigma
    return (mu - f_max) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)

def bootstrap_predict_strength(features, n_bootstrap=50):
    preds = []
    for i in range(n_bootstrap):
        bootstrap_indices = np.random.choice(len(X_strength_train), size=len(X_strength_train), replace=True)
        X_bootstrap = X_strength_scaled[bootstrap_indices]
        y_bootstrap = y_strength_train.iloc[bootstrap_indices]
        
        model = RandomForestRegressor(**RF_HYPERPARAMS_STRENGTH, random_state=42+i)
        model.fit(X_bootstrap, y_bootstrap)
        X_pred_scaled = strength_scaler.transform(np.array(features).reshape(1, -1))
        preds.append(model.predict(X_pred_scaled)[0])
    return np.mean(preds), np.std(preds)

def bootstrap_predict_strain(features, n_bootstrap=50):
    preds = []
    for i in range(n_bootstrap):
        bootstrap_indices = np.random.choice(len(X_strain_train), size=len(X_strain_train), replace=True)
        X_bootstrap = X_strain_scaled[bootstrap_indices]
        y_bootstrap = y_strain_train.iloc[bootstrap_indices]
        
        model = RandomForestRegressor(**RF_HYPERPARAMS_STRAIN, random_state=42+i)
        model.fit(X_bootstrap, y_bootstrap)
        X_pred_scaled = strain_scaler.transform(np.array(features).reshape(1, -1))
        preds.append(model.predict(X_pred_scaled)[0])
    return np.mean(preds), np.std(preds)

def process_single_alloy(alloy):
    try:
        strength_feats, strain_feats = calculate_alloy_features(alloy)
        if strength_feats is None or strain_feats is None:
            return None
        
        strength_mu, strength_sigma = bootstrap_predict_strength(strength_feats)
        strain_mu, strain_sigma = bootstrap_predict_strain(strain_feats)
        
        strength_ei = calculate_ei(strength_mu, strength_sigma, current_best_strength)
        strain_ei = calculate_ei(strain_mu, strain_sigma, current_best_strain)
        
        return {
            **{element: alloy[element] for element in selected_elements},
            'Strength_MPa': strength_mu, 'Strain_%': strain_mu,
            'Strength_std': strength_sigma, 'Strain_std': strain_sigma,
            'Strength_EI': strength_ei, 'Strain_EI': strain_ei,
            'Total_EI': strength_ei + strain_ei
        }
    except Exception:
        return None

def find_pareto_front(results_df, obj1='Strength_EI', obj2='Strain_EI'):
    def is_dominated(a, b):
        return (b[0] >= a[0] and b[1] >= a[1]) and (b[0] > a[0] or b[1] > a[1])
    
    properties_array = results_df[[obj1, obj2]].values
    pareto_indices = []
    for i in range(len(properties_array)):
        dominated = False
        for j in range(len(properties_array)):
            if i != j and is_dominated(properties_array[i], properties_array[j]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    
    pareto_df = results_df.iloc[pareto_indices].copy()
    pareto_front = properties_array[pareto_indices]
    return pareto_df, pareto_front

def find_optimal_clusters(pareto_front, max_clusters=10):
    wgss_values = []
    max_possible_k = min(max_clusters, len(pareto_front) - 1)
    k_range = range(1, max_possible_k + 1) if max_possible_k >= 1 else [1]
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pareto_front)
        wgss_values.append(kmeans.inertia_)
    
    optimal_k = 1
    if len(wgss_values) > 2:
        second_deriv = np.diff(wgss_values, 2)
        optimal_k = np.argmin(second_deriv) + 2
    elif len(wgss_values) == 2:
        optimal_k = 2
    
    return optimal_k, wgss_values

def perform_clustering(pareto_front, k_value):
    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pareto_front)
    cluster_centers = kmeans.cluster_centers_
    
    best_indices = []
    for cluster_id in range(k_value):
        cluster_point_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_points = pareto_front[cluster_point_indices]
        center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        closest_local_idx = np.argmin(distances)
        best_indices.append(cluster_point_indices[closest_local_idx])
    
    return best_indices, cluster_labels, cluster_centers

def main():
    print("="*60)
    print("RHEA Strength-Ductility Optimization")
    print("="*60)
    
    total_samples = 1000000
    print(f"Generating {total_samples} virtual alloy samples using Latin Hypercube Sampling...")
    
    virtual_alloys = latin_hypercube_sampling(total_samples, selected_elements)
    
    ei_save_path = os.path.join(intermediate_path, "virtual_alloy_ei_results.joblib")
    
    if os.path.exists(ei_save_path):
        user_choice = input("EI results exist! Recalculate? (y/N): ").strip().lower()
        if user_choice != 'y':
            print("Using existing EI results.")
            ei_results_df = joblib.load(ei_save_path)
        else:
            print("Recalculating EI values...")
            ei_results = []
            for alloy in tqdm(virtual_alloys, desc="Processing alloys"):
                res = process_single_alloy(alloy)
                if res is not None:
                    ei_results.append(res)
            
            ei_results_df = pd.DataFrame(ei_results)
            joblib.dump(ei_results_df, ei_save_path)
    else:
        print("Calculating EI values...")
        ei_results = []
        for alloy in tqdm(virtual_alloys, desc="Processing alloys"):
            res = process_single_alloy(alloy)
            if res is not None:
                ei_results.append(res)
        
        ei_results_df = pd.DataFrame(ei_results)
        joblib.dump(ei_results_df, ei_save_path)
    
    print(f"Valid results: {len(ei_results_df)}/{total_samples}")
    
    pareto_df, pareto_front = find_pareto_front(ei_results_df)
    print(f"Pareto front points: {len(pareto_front)}")
    
    optimal_k, wgss_values = find_optimal_clusters(pareto_front)
    print(f"Suggested K value: {optimal_k}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(wgss_values)+1), wgss_values, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
    plt.ylabel('Within-Cluster Sum of Squares', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Optimal Cluster Number', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3, linestyle='--')
    
    if 1 <= optimal_k <= len(wgss_values):
        plt.scatter(optimal_k, wgss_values[optimal_k-1], color='red', s=100, zorder=5)
        plt.annotate(f'K={optimal_k}', xy=(optimal_k, wgss_values[optimal_k-1]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')
    
    elbow_save_path = os.path.join(desktop_path, 'elbow_method.png')
    plt.savefig(elbow_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    selected_k = optimal_k
    best_indices, cluster_labels, cluster_centers = perform_clustering(pareto_front, selected_k)
    best_alloys_df = pareto_df.iloc[best_indices].copy()
    best_alloys_df['Cluster_ID'] = cluster_labels[best_indices]
    
    final_save_path = os.path.join(desktop_path, 'RHEA_optimization_results.xlsx')
    with pd.ExcelWriter(final_save_path, engine='openpyxl') as writer:
        ei_results_df.to_excel(writer, sheet_name='All_EI_Results', index=False)
        pareto_df.to_excel(writer, sheet_name='Pareto_Front', index=False)
        best_alloys_df.to_excel(writer, sheet_name='Optimal_Alloys', index=False)
    
    print(f"\nOptimal RHEA Compositions (K={selected_k}):")
    print("="*80)
    for idx, (_, row) in enumerate(best_alloys_df.iterrows(), 1):
        composition_str = ""
        for element in selected_elements:
            if row[element] > 0:
                composition_str += f"{element}~{int(round(row[element]))}~"
        print(f"Candidate {idx} (Cluster {row['Cluster_ID']}):")
        print(f"  Composition: {composition_str}")
        print(f"  Strength: {row['Strength_MPa']:.1f} ± {row['Strength_std']:.1f} MPa")
        print(f"  Strain: {row['Strain_%']:.1f} ± {row['Strain_std']:.1f} %")
        print(f"  EI Values: Strength {row['Strength_EI']:.4f} | Strain {row['Strain_EI']:.4f}")
        print("-" * 80)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(ei_results_df['Strength_EI'], ei_results_df['Strain_EI'], 
                color='#e0e0e0', alpha=0.5, s=20, label='All Alloys', zorder=1)
    plt.scatter(pareto_df['Strength_EI'], pareto_df['Strain_EI'], 
                color='#808080', s=40, label='Pareto Front', zorder=2)
    plt.scatter(best_alloys_df['Strength_EI'], best_alloys_df['Strain_EI'], 
                color='red', s=120, marker='o', edgecolors='black', linewidth=1.5, 
                label='Optimal Alloys', zorder=3)
    
    for i, (_, row) in enumerate(best_alloys_df.iterrows()):
        plt.annotate(f'Opt{i+1}', xy=(row['Strength_EI'], row['Strain_EI']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.xlabel('Strength Expected Improvement (EI)', fontsize=12, fontweight='bold')
    plt.ylabel('Strain Expected Improvement (EI)', fontsize=12, fontweight='bold')
    plt.title('EI Pareto Front & Optimal Alloys', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    pareto_save_path = os.path.join(desktop_path, 'ei_pareto_front.png')
    plt.savefig(pareto_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {final_save_path}")
    print(f"Charts saved to desktop")
    print("Optimization completed successfully!")

if __name__ == '__main__':
    main()
