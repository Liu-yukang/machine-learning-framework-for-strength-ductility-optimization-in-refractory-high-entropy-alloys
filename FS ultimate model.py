import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from typing import Tuple, List
import joblib
import os

# --------------------
# Configuration
# --------------------
plt.rcParams["font.family"] = ["Times New Roman", "Arial"]
RANDOM_SEED = 42

# Outlier handling params (IQR method)
OUTLIER_PARAMS = {
    "use_iqr": True,
    "iqr_threshold": 1.5
}

# Random Forest hyperparams
RF_HYPERPARAMS = {
    'bootstrap': True, 
    'ccp_alpha': 0.01, 
    'max_depth': 15, 
    'max_features': 'sqrt', 
    'min_impurity_decrease': 0.0, 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'n_estimators': 213
}

# --------------------
# DataProcessor Class
# --------------------
class DataProcessor:
    def __init__(self, file_path: str, features: List[str], target: str,
                 use_iqr: bool = True, iqr_threshold: float = 1.5):
        self.file_path = file_path
        self.features = features
        self.target = target
        self.use_iqr = use_iqr
        self.iqr_threshold = iqr_threshold
        self.df = None
        self.X = None
        self.y = None

    def load_data(self) -> None:
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Data loaded successfully, shape: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: File not found. Check path\nPath: {self.file_path}")
        except Exception as e:
            raise Exception(f"Data loading failed: {str(e)}")

    def remove_outliers(self) -> None:
        if not self.use_iqr:
            return
            
        combined = pd.concat([self.df[self.features], self.df[self.target]], axis=1)
        original_count = len(combined)
        
        Q1 = combined.quantile(0.25)
        Q3 = combined.quantile(0.75)
        IQR = Q3 - Q1
        
        mask = ~((combined < (Q1 - self.iqr_threshold * IQR)) | 
                 (combined > (Q3 + self.iqr_threshold * IQR))).any(axis=1)
        
        self.df = self.df[mask].copy()
        removed_count = original_count - len(self.df)
        print(f"IQR outlier handling: Removed {removed_count} outlier samples, remaining {len(self.df)} samples")

    def preprocess(self) -> None:
        if self.df is None:
            raise ValueError("Call load_data() first to load data")
        
        # Handle missing values
        self.df = self.df.dropna(subset=self.features + [self.target])
        print(f"After missing value handling: {self.df.shape[0]} samples")
        
        # Handle outliers
        self.remove_outliers()
        
        # Extract features and target
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        print(f"Final data after preprocessing: {self.df.shape[0]} samples, {len(self.features)} features")

# --------------------
# DataAugmenter Class
# --------------------
class DataAugmenter:
    @staticmethod
    def add_zscore_noise(X: pd.DataFrame, noise_level: float = 0.05) -> pd.DataFrame:
        if noise_level < 0:
            raise ValueError("noise_level must be non-negative")
        
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        noise = np.random.normal(0, 1, X.shape) * stds.values * noise_level
        return X + noise

    @classmethod
    def augment_data(cls, X: pd.DataFrame, y: pd.Series, 
                    n_augmentations: int = 3, noise_level: float = 0.03) -> Tuple[pd.DataFrame, pd.Series]:
        if n_augmentations < 0:
            raise ValueError("n_augmentations must be non-negative integer")
        
        X_augmented = [X]
        y_augmented = [y]
        
        for _ in range(n_augmentations):
            X_noisy = cls.add_zscore_noise(X, noise_level)
            X_augmented.append(X_noisy)
            y_augmented.append(y)
        
        return pd.concat(X_augmented, axis=0), pd.concat(y_augmented, axis=0)

# --------------------
# FeatureSelector Class
# --------------------
class FeatureSelector:
    @staticmethod
    def l1_regularization(X: pd.DataFrame, y: pd.Series, alpha: float = 0.01) -> Tuple[np.ndarray, List[str], StandardScaler]:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lasso = Lasso(alpha=alpha, random_state=RANDOM_SEED)
        lasso.fit(X_scaled, y)
        
        selector = SelectFromModel(lasso, prefit=True)
        X_selected = selector.transform(X_scaled)
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()
        
        print(f"L1 regularization feature selection: {len(selected_features)}/{len(X.columns)}")
        print(f"Retained features: {', '.join(selected_features)}")
        return X_selected, selected_features, scaler

# --------------------
# ModelTrainer Class
# --------------------
class ModelTrainer:
    @staticmethod
    def train_random_forest(X: np.ndarray, y: pd.Series) -> RandomForestRegressor:
        model = RandomForestRegressor(
            **RF_HYPERPARAMS,
            random_state=RANDOM_SEED
        )
        model.fit(X, y)
        return model

    @staticmethod
    def cross_validate(X: np.ndarray, y: pd.Series, model: RandomForestRegressor,
                       n_splits: int = 5) -> Tuple[List[float], List[float]]:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        train_scores = []
        test_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_train_pred)
            train_scores.append(train_r2)
            
            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)
            test_scores.append(test_r2)
        
        return train_scores, test_scores

    @staticmethod
    def evaluate_model(model: RandomForestRegressor, X: np.ndarray, y: pd.Series) -> dict:
        y_pred = model.predict(X)
        
        # Calculate MAPE (handle y=0 to avoid division by zero)
        non_zero_mask = y != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y[non_zero_mask] - y_pred[non_zero_mask]) / y[non_zero_mask])) * 100
        else:
            mape = np.nan
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': mape
        }
        
        return metrics

    @staticmethod
    def plot_feature_importance(model: RandomForestRegressor, feature_names: List[str], 
                               top_n: int = None, figsize: Tuple[int, int] = (10, 6)) -> None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        if top_n is None:
            top_n = len(feature_names)
        indices = indices[:top_n]
        
        plt.figure(figsize=figsize)
        plt.title("Feature Importance Ranking", fontsize=14)
        plt.bar(range(top_n), importances[indices], align="center")
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.xlim([-1, top_n])
        plt.tight_layout()
        plt.show()
        
        print("\nFeature Importance Ranking:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    @staticmethod
    def plot_prediction_vs_actual(model: RandomForestRegressor, X: np.ndarray, y: pd.Series,
                                 title: str = "Predicted vs Actual Values", figsize: Tuple[int, int] = (10, 8)) -> None:
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        plt.figure(figsize=figsize)
        
        plt.subplot(2, 1, 1)
        plt.scatter(y, y_pred, alpha=0.7, edgecolors='w')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Predicted', fontsize=12)
        plt.title(f'{title} - Scatter Plot', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.subplot(2, 1, 2)
        plt.scatter(y_pred, residuals, alpha=0.7, edgecolors='w')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title(f'{title} - Residual Plot', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        metrics = ModelTrainer.evaluate_model(model, X, y)
        print(f"\nModel Evaluation Metrics:")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "MAPE: Meaningless (y=0 exists)")

    @staticmethod
    def plot_train_test_comparison(train_metrics: dict, test_metrics: dict, figsize: Tuple[int, int] = (12, 6)) -> None:
        metrics = ['r2', 'rmse', 'mae', 'mape']
        metric_names = ['R²', 'RMSE', 'MAE', 'MAPE(%)']
        
        train_values = [train_metrics[m] for m in metrics]
        test_values = [test_metrics[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=figsize)
        plt.bar(x - width/2, train_values, width, label='Train Set')
        plt.bar(x + width/2, test_values, width, label='Test Set')
        
        plt.ylabel('Metric Value')
        plt.title('Train vs Test Set Performance Comparison', fontsize=14)
        plt.xticks(x, metric_names)
        plt.legend()
        
        # Add value labels
        for i, (train_val, test_val) in enumerate(zip(train_values, test_values)):
            if metrics[i] == 'mape':
                plt.text(i - width/2, train_val + 0.01, f'{train_val:.2f}%', ha='center', va='bottom')
                plt.text(i + width/2, test_val + 0.01, f'{test_val:.2f}%', ha='center', va='bottom')
            else:
                plt.text(i - width/2, train_val + 0.01, f'{train_val:.4f}', ha='center', va='bottom')
                plt.text(i + width/2, test_val + 0.01, f'{test_val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# --------------------
# ModelPredictor Class
# --------------------
class ModelPredictor:
    def __init__(self, model_path: str, scaler_path: str, feature_names: List[str]):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = feature_names
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        missing_features = [f for f in self.feature_names if f not in new_data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        X = new_data[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_from_dict(self, data_dict: dict) -> np.ndarray:
        df = pd.DataFrame([data_dict])
        return self.predict(df)

# --------------------
# Main Workflow
# --------------------
def main():
    # Configuration params
    file_path = r'C:\Users\Liu\Desktop\FS-YS-over.xlsx'
    features = ['D.r', 'Hmix', 'Λ', 'ƞ', 'a']
    target = 'Ys'
    use_l1 = True
    l1_alpha = 0.01
    n_repeats = 5
    n_augmentations = 3
    noise_level = 0.03
    use_iqr = OUTLIER_PARAMS["use_iqr"]
    iqr_threshold = OUTLIER_PARAMS["iqr_threshold"]

    # 1. Data processing
    processor = DataProcessor(
        file_path, 
        features, 
        target,
        use_iqr=use_iqr,
        iqr_threshold=iqr_threshold
    )
    try:
        processor.load_data()
        processor.preprocess()
    except Exception as e:
        print(f"Data processing failed: {str(e)}")
        return None, None, None

    # 2. Data augmentation
    X_aug, y_aug = DataAugmenter.augment_data(
        processor.X, processor.y,
        n_augmentations=n_augmentations,
        noise_level=noise_level
    )
    print(f"Data after augmentation: {len(y_aug)} samples")

    # 3. Feature selection
    if use_l1:
        try:
            X_processed, selected_features, scaler = FeatureSelector.l1_regularization(X_aug, y_aug, l1_alpha)
        except Exception as e:
            print(f"Feature selection failed: {str(e)}")
            return None, None, None
    else:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X_aug)
        selected_features = features
        print("No feature selection used, all features retained")

    # 4. Model training & cross-validation
    train_r2_scores_all = []
    test_r2_scores_all = []
    best_model = None
    best_test_score = -float('inf')
    
    for repeat in range(1, n_repeats+1):
        model = ModelTrainer.train_random_forest(X_processed, y_aug)
        train_scores, test_scores = ModelTrainer.cross_validate(X_processed, y_aug, model)
        
        train_r2_scores_all.extend(train_scores)
        test_r2_scores_all.extend(test_scores)
        
        train_mean = np.mean(train_scores)
        test_mean = np.mean(test_scores)
        
        print(f"Experiment {repeat}/{n_repeats} completed:")
        print(f"  Train set average R²: {train_mean:.4f}")
        print(f"  Test set average R²: {test_mean:.4f}")
        
        if test_mean > best_test_score:
            best_test_score = test_mean
            best_model = model

    # 5. Result statistics
    train_mean_r2 = np.mean(train_r2_scores_all)
    train_std_r2 = np.std(train_r2_scores_all)
    test_mean_r2 = np.mean(test_r2_scores_all)
    test_std_r2 = np.std(test_r2_scores_all)
    
    print(f"\n===== Cross-Validation Results Summary =====")
    print(f"Train set R²: {train_mean_r2:.4f} ± {train_std_r2:.4f}")
    print(f"Test set R²: {test_mean_r2:.4f} ± {test_std_r2:.4f}")
    
    # 6. Evaluate final model
    print("\n===== Final Model Evaluation =====")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_aug, test_size=0.2, random_state=RANDOM_SEED
    )
    
    best_model.fit(X_train, y_train)
    train_metrics = ModelTrainer.evaluate_model(best_model, X_train, y_train)
    test_metrics = ModelTrainer.evaluate_model(best_model, X_test, y_test)
    
    print("\nTrain Set Evaluation:")
    for metric, value in train_metrics.items():
        if metric == 'mape':
            print(f"  {metric.upper()}: {value:.2f}%" if not np.isnan(value) else "  MAPE: Meaningless (y=0 exists)")
        else:
            print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTest Set Evaluation:")
    for metric, value in test_metrics.items():
        if metric == 'mape':
            print(f"  {metric.upper()}: {value:.2f}%" if not np.isnan(value) else "  MAPE: Meaningless (y=0 exists)")
        else:
            print(f"  {metric.upper()}: {value:.4f}")
    
    # 7. Visualizations
    ModelTrainer.plot_train_test_comparison(train_metrics, test_metrics)
    ModelTrainer.plot_feature_importance(
        best_model, 
        selected_features, 
        top_n=min(10, len(selected_features))
    )
    ModelTrainer.plot_prediction_vs_actual(
        best_model, 
        X_test, 
        y_test, 
        title="Test Set Prediction Evaluation"
    )
    
    # 8. Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/strength_model.pkl')
    joblib.dump(scaler, 'models/strength_scaler.pkl')
    joblib.dump(selected_features, 'models/selected_features_strength.pkl')
    print("\nModel, scaler, and feature names saved successfully!")
    
    # 9. Prediction example
    print("\n===== Prediction Example =====")
    try:
        sample_data = processor.X.head(5)
        actual_values = processor.y.head(5).values
        predictions = best_model.predict(scaler.transform(sample_data))
        
        print("\nSample Prediction Results:")
        for i, (actual, pred) in enumerate(zip(actual_values, predictions)):
            print(f"Sample {i+1}: Actual={actual:.4f}, Predicted={pred:.4f}, Error={pred-actual:.4f}")
    except Exception as e:
        print(f"Prediction example failed: {str(e)}")
    
    return best_model, scaler, selected_features

# --------------------
# Prediction Script Example
# --------------------
if __name__ == "__main__":
    print("\nSelect Mode:")
    print("1. Train Model")
    print("2. Predict with Trained Model")
    choice = input("Enter 1/2: ").strip()
    
    if choice == "1":
        best_model, scaler, features = main()
        
    elif choice == "2":
        try:
            model_path = 'models/strength_model.pkl'
            scaler_path = 'models/strength_scaler.pkl'
            features_path = 'models/selected_features_strength.pkl'
            
            if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
                print("Model files not found. Train first by running Mode 1!")
            else:
                selected_features = joblib.load(features_path)
                predictor = ModelPredictor(model_path, scaler_path, selected_features)
                
                new_data = pd.DataFrame({
                    'D.r': [0.85, 0.90],
                    'Hmix': [-15, -20],
                    'Λ': [1.2, 1.5],
                    'ƞ': [0.8, 0.9],
                    'a': [3.5, 3.6]
                })
                
                predictions = predictor.predict(new_data)
                print("\nNew Data Prediction Results:")
                for i, pred in enumerate(predictions):
                    print(f"Sample {i+1}: Predicted={pred:.4f}")
                    
                # Single sample prediction
                single_prediction = predictor.predict_from_dict({
                    'D.r': 0.88,
                    'Hmix': -18,
                    'Λ': 1.3,
                    'ƞ': 0.85,
                    'a': 3.55
                })
                print(f"\nSingle Sample Prediction: {single_prediction[0]:.4f}")
                
        except Exception as e:
            print(f"Prediction process error: {str(e)}")
    
    else:
        print("Invalid choice. Re-run and enter 1 or 2.")
