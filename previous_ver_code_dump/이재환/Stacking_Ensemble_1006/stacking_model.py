import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LogisticRegression, Ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import json

from config import Config


class StackingEnsembleWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.base_models = {}
        self.meta_model = None
        self.train_history = {'train_loss': [], 'val_loss': []}
        self.metrics_history = {}
        self.model_save_path = "stacking_ensemble_model.pkl"
        self.metadata_save_path = "stacking_metadata.pkl"
        self.label_encoders = {}
        self.feature_names = None

    def _preprocess_data(self, df):
        """데이터 전처리: 타입 변환 및 정리"""
        df = df.copy()

        for col in df.columns:
            # 1. 결측값 처리
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object', 'category']:
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())

            # 2. 데이터 타입 자동 감지 및 변환
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    df[col] = pd.to_numeric(df[col])
                    print(f"Converted {col} to numeric")
                except:
                    unique_ratio = df[col].nunique() / len(df)

                    if unique_ratio > 0.9 and df[col].nunique() > 100:
                        print(f"Warning: Column '{col}' has too many unique values ({df[col].nunique()}), might be ID column")
                        top_categories = df[col].value_counts().head(50).index
                        df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')
                        print(f"-> Reduced to top 50 categories + 'Other'")

                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))

                    print(f"✓ Processed {col} as categorical ({df[col].nunique()} categories)")

            elif df[col].dtype in ['int64', 'float64']:
                if df[col].max() > 1e6:
                    print(f"Info: Large values detected in {col} (max: {df[col].max():.2e})")

        return df

    def prepare_data(self, df, target_columns, test_size=0.2, val_size=0.1):
        if 'ENCODED_MCT' in df.columns:
            print("ENCODED_MCT 기준으로 데이터셋 분리")
            return self._prepare_data_by_store(df, target_columns, test_size, val_size)
        else:
            print("일반적인 방법으로 데이터셋 분리")
            return self._prepare_data_standard(df, target_columns, test_size, val_size)

    def _prepare_data_by_store(self, df, target_columns, test_size=0.2, val_size=0.1):
        """ENCODED_MCT 기준으로 데이터셋을 분리 (점포별로 분리)"""
        X = df.drop(columns=target_columns)
        y = df[target_columns].values

        if 'TA_YM' in df.columns:
            max_ta_ym_per_store = df.groupby('ENCODED_MCT')['TA_YM'].max().reset_index()
            max_ta_ym_per_store.columns = ['ENCODED_MCT', 'max_TA_YM']

            df_with_max = df.merge(max_ta_ym_per_store, on='ENCODED_MCT')
            test_indices = df_with_max[df_with_max['TA_YM'] == df_with_max['max_TA_YM']].index

            X_test = X.iloc[test_indices].copy()
            y_test = y[test_indices]

            remaining_indices = df.index.difference(test_indices)
            X_remaining = X.iloc[remaining_indices].copy()
            y_remaining = y[remaining_indices]

            print(f"Test set: {X_test.shape} (각 점포의 최신 데이터)")
            print(f"Remaining data: {X_remaining.shape}")

            # 나머지 데이터를 점포 기준으로 train/validation 분리
            unique_stores = X_remaining['ENCODED_MCT'].unique()
            val_ratio = val_size / (1 - test_size)
            train_stores, val_stores = train_test_split(
                unique_stores, test_size=val_ratio, random_state=42
            )

            # 점포별로 데이터 분리
            train_mask = X_remaining['ENCODED_MCT'].isin(train_stores)
            X_train = X_remaining[train_mask].copy()
            y_train = y_remaining[train_mask]

            val_mask = X_remaining['ENCODED_MCT'].isin(val_stores)
            X_val = X_remaining[val_mask].copy()
            y_val = y_remaining[val_mask]

            print(f"Train set: {X_train.shape} ({len(train_stores)} 점포)")
            print(f"Validation set: {X_val.shape} ({len(val_stores)} 점포)")

        else:
            # TA_YM이 없으면 점포별로만 분리
            unique_stores = X['ENCODED_MCT'].unique()

            # 점포를 train/val/test로 분리
            test_stores, temp_stores = train_test_split(
                unique_stores, test_size=test_size, random_state=42
            )
            val_ratio = val_size / (1 - test_size)
            train_stores, val_stores = train_test_split(
                temp_stores, test_size=val_ratio, random_state=42
            )

            # 각 점포별로 데이터 분리
            X_train = X[X['ENCODED_MCT'].isin(train_stores)].copy()
            X_val = X[X['ENCODED_MCT'].isin(val_stores)].copy()
            X_test = X[X['ENCODED_MCT'].isin(test_stores)].copy()

            y_train = y[X['ENCODED_MCT'].isin(train_stores)]
            y_val = y[X['ENCODED_MCT'].isin(val_stores)]
            y_test = y[X['ENCODED_MCT'].isin(test_stores)]

        return self._finalize_data_preparation(X, X_train, X_val, X_test, y_train, y_val, y_test)

    def _prepare_data_standard(self, df, target_columns, test_size=0.2, val_size=0.1):
        """일반적인 방법으로 데이터셋 분리"""
        X = df.drop(columns=target_columns)
        y = df[target_columns].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42
        )

        return self._finalize_data_preparation(X, X_train, X_val, X_test, y_train, y_val, y_test)

    def _finalize_data_preparation(self, X, X_train, X_val, X_test, y_train, y_val, y_test):
        # 데이터 전처리 및 타입 정리
        X_processed = self._preprocess_data(X)

        # 전처리된 데이터로 다시 분할
        X_train_processed = X_processed.iloc[X_train.index]
        X_val_processed = X_processed.iloc[X_val.index]
        X_test_processed = X_processed.iloc[X_test.index]

        # Feature names 저장 및 특수문자 제거 (LightGBM 호환성)
        self.feature_names = X_train_processed.columns.tolist()
        self.feature_names_clean = [
            col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
               .replace('"', '').replace(',', '_').replace(':', '_').replace(' ', '')
            for col in self.feature_names
        ]

        # 컬럼명 변경
        X_train_processed.columns = self.feature_names_clean
        X_val_processed.columns = self.feature_names_clean
        X_test_processed.columns = self.feature_names_clean

        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

    def build_model(self, input_dim, output_dim=1):
        """스태킹 앙상블 모델 빌드 (Base Models + Meta Model)"""
        device = self.config.model.xgb_device
        if device == 'mps':
            device = 'cpu'
            print("Warning: Models do not support MPS. Using CPU instead.")

        # Base Models
        if self.config.model.task == 'regression':
            self.base_models['xgboost'] = xgb.XGBRegressor(
                n_estimators=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                max_depth=self.config.model.xgb_max_depth,
                min_child_weight=self.config.model.xgb_min_child_weight,
                subsample=self.config.model.xgb_subsample,
                colsample_bytree=self.config.model.xgb_colsample_bytree,
                gamma=self.config.model.xgb_gamma,
                reg_alpha=self.config.model.xgb_reg_alpha,
                reg_lambda=self.config.model.xgb_reg_lambda,
                random_state=42,
                tree_method=self.config.model.xgb_tree_method,
                device=device,
                eval_metric=self.config.model.xgb_eval_metric
            )

            self.base_models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                num_leaves=self.config.model.lgbm_num_leaves,
                max_depth=self.config.model.lgbm_max_depth,
                min_child_samples=self.config.model.lgbm_min_child_samples,
                subsample=self.config.model.lgbm_subsample,
                colsample_bytree=self.config.model.lgbm_colsample_bytree,
                reg_alpha=self.config.model.lgbm_reg_alpha,
                reg_lambda=self.config.model.lgbm_reg_lambda,
                random_state=42,
                verbose=-1
            )

            self.base_models['catboost'] = CatBoostRegressor(
                iterations=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                depth=self.config.model.catboost_depth,
                l2_leaf_reg=self.config.model.catboost_l2_leaf_reg,
                border_count=self.config.model.catboost_border_count,
                random_state=42,
                verbose=0
            )

            # Meta Model
            if self.config.model.meta_model_type == 'xgboost':
                self.meta_model = xgb.XGBRegressor(
                    n_estimators=self.config.model.meta_model_epochs,
                    learning_rate=self.config.model.meta_model_lr,
                    max_depth=self.config.model.meta_model_max_depth,
                    random_state=42,
                    device=device
                )
            else:
                self.meta_model = Ridge(alpha=1.0, random_state=42)

        else:
            self.base_models['xgboost'] = xgb.XGBClassifier(
                n_estimators=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                max_depth=self.config.model.xgb_max_depth,
                min_child_weight=self.config.model.xgb_min_child_weight,
                subsample=self.config.model.xgb_subsample,
                colsample_bytree=self.config.model.xgb_colsample_bytree,
                gamma=self.config.model.xgb_gamma,
                reg_alpha=self.config.model.xgb_reg_alpha,
                reg_lambda=self.config.model.xgb_reg_lambda,
                scale_pos_weight=self.config.model.xgb_scale_pos_weight,
                random_state=42,
                tree_method=self.config.model.xgb_tree_method,
                device=device,
                eval_metric='logloss'
            )

            self.base_models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                num_leaves=self.config.model.lgbm_num_leaves,
                max_depth=self.config.model.lgbm_max_depth,
                min_child_samples=self.config.model.lgbm_min_child_samples,
                subsample=self.config.model.lgbm_subsample,
                colsample_bytree=self.config.model.lgbm_colsample_bytree,
                reg_alpha=self.config.model.lgbm_reg_alpha,
                reg_lambda=self.config.model.lgbm_reg_lambda,
                random_state=42,
                verbose=-1
            )

            self.base_models['catboost'] = CatBoostClassifier(
                iterations=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                depth=self.config.model.catboost_depth,
                l2_leaf_reg=self.config.model.catboost_l2_leaf_reg,
                border_count=self.config.model.catboost_border_count,
                random_state=42,
                verbose=0
            )

            # Meta Model
            if self.config.model.meta_model_type == 'xgboost':
                self.meta_model = xgb.XGBClassifier(
                    n_estimators=self.config.model.meta_model_epochs,
                    learning_rate=self.config.model.meta_model_lr,
                    max_depth=self.config.model.meta_model_max_depth,
                    random_state=42,
                    device=device
                )
            else:
                self.meta_model = LogisticRegression(max_iter=1000, random_state=42)

        print(f"Stacking Ensemble model built for {self.config.model.task}")
        print(f"Base Models: {list(self.base_models.keys())}")
        print(f"Meta Model: {self.config.model.meta_model_type}")

    def train_model(self, X_train, y_train, X_val, y_val):
        """스태킹 앙상블 모델 학습"""
        if len(y_train.shape) == 2 and y_train.shape[1] == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()

        # Base Models 학습
        print("Training Base Models...")
        train_preds = np.zeros((X_train.shape[0], len(self.base_models)))
        val_preds = np.zeros((X_val.shape[0], len(self.base_models)))

        for idx, (name, model) in enumerate(self.base_models.items()):
            print(f"  Training {name}...")
            if name == 'xgboost':
                verbose_flag = 100 if self.config.model.verbose > 0 else 0
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=verbose_flag
                )
            elif name == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.log_evaluation(period=100)] if self.config.model.verbose > 0 else [lgb.log_evaluation(period=0)]
                )
            elif name == 'catboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=100 if self.config.model.verbose > 0 else 0
                )
            else:
                model.fit(X_train, y_train)
                if self.config.model.verbose > 0:
                    print(f"    {name} training in progress...")

            train_preds[:, idx] = model.predict(X_train)
            val_preds[:, idx] = model.predict(X_val)
            print(f"  {name} trained")

        # Meta Model 학습
        print("Training Meta Model...")
        if isinstance(self.meta_model, (xgb.XGBRegressor, xgb.XGBClassifier)):
            self.meta_model.fit(
                train_preds, y_train,
                eval_set=[(train_preds, y_train), (val_preds, y_val)],
                verbose=self.config.model.verbose
            )
            results = self.meta_model.evals_result()
            if self.config.model.task == 'regression':
                self.train_history['train_loss'] = results['validation_0'].get('rmse', [])
                self.train_history['val_loss'] = results['validation_1'].get('rmse', [])
            else:
                self.train_history['train_loss'] = results['validation_0'].get('logloss', [])
                self.train_history['val_loss'] = results['validation_1'].get('logloss', [])
        else:
            self.meta_model.fit(train_preds, y_train)
            train_pred_final = self.meta_model.predict(train_preds)
            val_pred_final = self.meta_model.predict(val_preds)

            if self.config.model.task == 'regression':
                train_loss = np.sqrt(mean_squared_error(y_train, train_pred_final))
                val_loss = np.sqrt(mean_squared_error(y_val, val_pred_final))
            else:
                from sklearn.metrics import log_loss
                train_loss = log_loss(y_train, train_pred_final)
                val_loss = log_loss(y_val, val_pred_final)

            self.train_history['train_loss'] = [train_loss]
            self.train_history['val_loss'] = [val_loss]

        print("Meta Model trained")

    def predict(self, X):
        """예측 수행"""
        X_processed = self._preprocess_data(X)

        # 컬럼명 변경 (LightGBM 호환성)
        if hasattr(self, 'feature_names_clean'):
            X_processed.columns = self.feature_names_clean

        # Base Models 예측
        base_preds = np.zeros((X_processed.shape[0], len(self.base_models)))
        for idx, (name, model) in enumerate(self.base_models.items()):
            base_preds[:, idx] = model.predict(X_processed)

        # Meta Model 예측
        if self.config.model.task == 'regression':
            predictions = self.meta_model.predict(base_preds)
            return predictions, None
        else:
            predictions = self.meta_model.predict(base_preds)
            probabilities = self.meta_model.predict_proba(base_preds) if hasattr(self.meta_model, 'predict_proba') else None
            return predictions, probabilities

    def evaluate_model(self, X_test, y_test):
        """모델 평가"""
        predictions, probabilities = self.predict(X_test)

        if len(y_test.shape) == 2 and y_test.shape[1] == 1:
            y_test = y_test.flatten()

        metrics = {}

        if self.config.model.task == 'regression':
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(y_test, predictions)
        else:
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            try:
                metrics['precision'] = precision_score(y_test, predictions, average='weighted')
                metrics['recall'] = recall_score(y_test, predictions, average='weighted')
                metrics['f1'] = f1_score(y_test, predictions, average='weighted')
            except:
                metrics['precision'] = 0.0
                metrics['recall'] = 0.0
                metrics['f1'] = 0.0

        self.metrics_history = metrics

        if self.config.model.verbose:
            print("📊 Model Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")

        return metrics, predictions, probabilities

    def get_feature_importance(self, X):
        """Feature importance 계산 (XGBoost base model 사용)"""
        if 'xgboost' not in self.base_models:
            return None

        importance = self.base_models['xgboost'].feature_importances_
        return importance

    def plot_training_history(self):
        """학습 히스토리 플롯"""
        if not os.path.exists(self.config.ui.plots_save_dir):
            os.makedirs(self.config.ui.plots_save_dir)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 학습 곡선
        if self.train_history['train_loss'] and self.train_history['val_loss']:
            ax1.plot(self.train_history['train_loss'], label='Training Loss')
            ax1.plot(self.train_history['val_loss'], label='Validation Loss')
            ax1.set_title('Meta Model Loss')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.legend()

        # 메트릭 막대그래프
        if self.metrics_history:
            metrics_names = list(self.metrics_history.keys())
            metrics_values = list(self.metrics_history.values())

            ax2.bar(metrics_names, metrics_values)
            ax2.set_title('Model Metrics')
            ax2.set_ylabel('Score')

            for i, v in enumerate(metrics_values):
                ax2.text(i, v + max(metrics_values) * 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()

        if self.config.ui.save_plots_image:
            plt.savefig(f"{self.config.ui.plots_save_dir}/stacking_ensemble_results.png", dpi=300, bbox_inches='tight')

        if self.config.ui.show_plots_popup:
            plt.show()
        else:
            plt.close()

        return fig

    def save_model(self, save_dir="./models"):
        """모델 저장"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, "stacking_ensemble_model.pkl")
        metadata_path = os.path.join(save_dir, "stacking_metadata.pkl")

        if self.meta_model is None:
            print("Warning: No model to save!")
            return False

        try:
            # 전체 모델 저장
            model_data = {
                'base_models': self.base_models,
                'meta_model': self.meta_model
            }

            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            # 메타데이터 저장
            metadata = {
                'train_history': self.train_history,
                'metrics_history': self.metrics_history,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'feature_names_clean': self.feature_names_clean if hasattr(self, 'feature_names_clean') else None,
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            print(f"Model saved to {model_path}")
            print(f"Metadata saved to {metadata_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, save_dir="./models"):
        """모델 로드"""
        model_path = os.path.join(save_dir, "stacking_ensemble_model.pkl")
        metadata_path = os.path.join(save_dir, "stacking_metadata.pkl")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"Error: Model files not found in {save_dir}")
            return False

        try:
            # 모델 로드
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.base_models = model_data['base_models']
            self.meta_model = model_data['meta_model']

            # 메타데이터 로드
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.train_history = metadata['train_history']
            self.metrics_history = metadata['metrics_history']
            self.label_encoders = metadata.get('label_encoders', {})
            self.feature_names = metadata.get('feature_names', None)
            self.feature_names_clean = metadata.get('feature_names_clean', None)

            print(f"Model loaded from {model_path}")
            print(f"Metadata loaded from {metadata_path}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def model_exists(self, save_dir="./models"):
        """모델 파일 존재 여부 확인"""
        model_path = os.path.join(save_dir, "stacking_ensemble_model.pkl")
        metadata_path = os.path.join(save_dir, "stacking_metadata.pkl")
        return os.path.exists(model_path) and os.path.exists(metadata_path)
