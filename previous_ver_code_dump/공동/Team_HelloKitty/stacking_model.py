import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
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
        # 베이스 모델들
        self.base_models = []
        # 메타 모델
        self.meta_model = None
        self.train_history = {'train_loss': [], 'val_loss': []}
        self.metrics_history = {}
        self.model_save_path = "stacking_model.pkl"
        self.metadata_save_path = "stacking_metadata.pkl"
        self.label_encoders = {}
        self.feature_names = None

    def _preprocess_data(self, df):
        df = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object', 'category']:
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())

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

                    print(f"Processed {col} as categorical ({df[col].nunique()} categories)")

            elif df[col].dtype in ['int64', 'float64']:
                if df[col].max() > 1e6:
                    print(f"Info: Large values detected in {col} (max: {df[col].max():.2e})")

        return df

    def _create_time_series_data(self, df, target_columns):
        """시계열 데이터 생성: N월 features로 N+1월 타겟 예측하도록 변환"""
        if 'ENCODED_MCT' not in df.columns or 'TA_YM' not in df.columns:
            print("Warning: ENCODED_MCT 또는 TA_YM 컬럼이 없어 시계열 변환을 건너뜁니다.")
            return df

        print("시계열 데이터 변환 중: N월 features → N+1월 target")

        # 점포별, 시간순 정렬
        df_sorted = df.sort_values(['ENCODED_MCT', 'TA_YM']).reset_index(drop=True)

        # 각 점포별로 타겟을 한 행 위로 shift (N+1월 타겟을 N월 행에 배치)
        df_sorted['next_month_target'] = df_sorted.groupby('ENCODED_MCT')[target_columns[0]].shift(-1)

        # 마지막 행(타겟이 없는 행) 제거
        df_shifted = df_sorted.dropna(subset=['next_month_target']).copy()

        # 원래 타겟 컬럼을 next_month_target으로 교체
        df_shifted[target_columns[0]] = df_shifted['next_month_target']
        df_shifted = df_shifted.drop(columns=['next_month_target'])

        print(f"시계열 변환 완료: {len(df)} → {len(df_shifted)} rows (마지막 월 데이터 제거)")
        print(f"예시: 202401월 features → 202402월 target 예측")

        return df_shifted

    def prepare_data(self, df, target_columns, test_size=0.2, val_size=0.1):
        # 시계열 데이터로 변환
        df = self._create_time_series_data(df, target_columns)

        if 'ENCODED_MCT' in df.columns:
            print("ENCODED_MCT 기준으로 데이터셋 분리")
            return self._prepare_data_by_store(df, target_columns, test_size, val_size)
        else:
            print("일반적인 방법으로 데이터셋 분리")
            return self._prepare_data_standard(df, target_columns, test_size, val_size)

    def _prepare_data_by_store(self, df, target_columns, test_size=0.2, val_size=0.1):
        """ENCODED_MCT 기준으로 데이터셋을 분리 (점포별로 분리)"""
        # 인덱스 리셋 (시계열 변환 후 인덱스가 변경되었을 수 있음)
        df = df.reset_index(drop=True)

        X = df.drop(columns=target_columns)
        y = df[target_columns].values

        if 'TA_YM' in df.columns:
            max_ta_ym_per_store = df.groupby('ENCODED_MCT')['TA_YM'].max().reset_index()
            max_ta_ym_per_store.columns = ['ENCODED_MCT', 'max_TA_YM']

            df_with_max = df.merge(max_ta_ym_per_store, on='ENCODED_MCT', how='left')
            test_mask = df_with_max['TA_YM'] == df_with_max['max_TA_YM']

            X_test = X[test_mask].copy()
            y_test = y[test_mask]

            remaining_mask = ~test_mask
            X_remaining = X[remaining_mask].copy()
            y_remaining = y[remaining_mask]

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

    def _preprocess_data_with_encoder(self, df):
        """학습된 label encoder를 사용하여 데이터 전처리 (Val/Test용)"""
        df = df.copy()

        for col in df.columns:
            # 1. 결측값 처리
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['object', 'category']:
                    df[col] = df[col].fillna('Unknown')
                else:
                    df[col] = df[col].fillna(df[col].median())

            # 2. 카테고리 데이터 처리 (학습된 encoder 사용)
            if df[col].dtype == 'object':
                if col in self.label_encoders:
                    # 새로운 카테고리는 'Unknown'으로 처리
                    unknown_mask = ~df[col].isin(self.label_encoders[col].classes_)
                    if unknown_mask.any():
                        df.loc[unknown_mask, col] = 'Unknown'
                        # Unknown이 encoder에 없으면 추가
                        if 'Unknown' not in self.label_encoders[col].classes_:
                            # 가장 빈도가 낮은 카테고리로 매핑
                            df.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]

                    df[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df

    def _finalize_data_preparation(self, X, X_train, X_val, X_test, y_train, y_val, y_test):
        # Train set으로 label encoder 학습
        X_train_processed = self._preprocess_data(X_train)

        # Val/Test set은 Train에서 학습한 encoder 사용
        X_val_processed = self._preprocess_data_with_encoder(X_val)
        X_test_processed = self._preprocess_data_with_encoder(X_test)

        # Feature names 저장
        self.feature_names = X_train_processed.columns.tolist()

        # Clean feature names for XGBoost compatibility
        self.feature_names_clean = [
            col.replace('[', '').replace(']', '').replace('<', '').replace('>', '')
               .replace('"', '').replace(',', '_').replace(':', '_').replace(' ', '')
            for col in self.feature_names
        ]

        X_train_processed.columns = self.feature_names_clean
        X_val_processed.columns = self.feature_names_clean
        X_test_processed.columns = self.feature_names_clean

        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

    def build_model(self, input_dim, output_dim=1):
        """베이스 모델 3개 (CatBoost, XGBoost, LightGBM)와 메타 모델 (XGBoost) 생성"""
        device = self.config.model.xgb_device
        if device == 'mps':
            device = 'cpu'
            print("Warning: XGBoost does not support MPS. Using CPU instead.")

        print("Building Stacking Ensemble Model...")
        print("Base Models: CatBoost, XGBoost, LightGBM")
        print("Meta Model: XGBoost")

        self.base_models = []

        if self.config.model.task == 'regression':
            # 베이스 모델 1: CatBoost
            catboost_model = CatBoostRegressor(
                iterations=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                depth=self.config.model.xgb_max_depth,
                l2_leaf_reg=self.config.model.xgb_reg_lambda,
                random_state=42,
                verbose=0,
                task_type='CPU'
            )
            self.base_models.append(('catboost', catboost_model))

            # 베이스 모델 2: XGBoost
            xgb_model = xgb.XGBRegressor(
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
                device=device
            )
            self.base_models.append(('xgboost', xgb_model))

            # 베이스 모델 3: LightGBM
            lgb_model = lgb.LGBMRegressor(
                n_estimators=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                max_depth=self.config.model.xgb_max_depth,
                num_leaves=31,
                subsample=self.config.model.xgb_subsample,
                colsample_bytree=self.config.model.xgb_colsample_bytree,
                reg_alpha=self.config.model.xgb_reg_alpha,
                reg_lambda=self.config.model.xgb_reg_lambda,
                random_state=42,
                verbose=-1,
                device='cpu'
            )
            self.base_models.append(('lightgbm', lgb_model))

            # 메타 모델: XGBoost
            self.meta_model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                tree_method=self.config.model.xgb_tree_method,
                device=device
            )
        else:
            # 베이스 모델 1: CatBoost
            catboost_model = CatBoostClassifier(
                iterations=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                depth=self.config.model.xgb_max_depth,
                l2_leaf_reg=self.config.model.xgb_reg_lambda,
                random_state=42,
                verbose=0,
                task_type='CPU'
            )
            self.base_models.append(('catboost', catboost_model))

            # 베이스 모델 2: XGBoost
            xgb_model = xgb.XGBClassifier(
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
                device=device
            )
            self.base_models.append(('xgboost', xgb_model))

            # 베이스 모델 3: LightGBM
            lgb_model = lgb.LGBMClassifier(
                n_estimators=self.config.model.epochs,
                learning_rate=self.config.model.lr,
                max_depth=self.config.model.xgb_max_depth,
                num_leaves=31,
                subsample=self.config.model.xgb_subsample,
                colsample_bytree=self.config.model.xgb_colsample_bytree,
                reg_alpha=self.config.model.xgb_reg_alpha,
                reg_lambda=self.config.model.xgb_reg_lambda,
                random_state=42,
                verbose=-1,
                device='cpu'
            )
            self.base_models.append(('lightgbm', lgb_model))

            # 메타 모델: XGBoost
            self.meta_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
                tree_method=self.config.model.xgb_tree_method,
                device=device
            )

        print(f"Stacking Ensemble Model built for {self.config.model.task}")
        print(f"  - Base models: {len(self.base_models)}")
        print(f"  - Meta model: XGBoost")

    def train_model(self, X_train, y_train, X_val, y_val):
        """2단계 스태킹 학습: 1) 베이스 모델 학습 2) 메타 모델 학습"""
        if len(y_train.shape) == 2 and y_train.shape[1] == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()

        print("\n" + "="*60)
        print("STAGE 1: Training Base Models")
        print("="*60)

        # 1단계: 베이스 모델들 학습
        base_train_predictions = []
        base_val_predictions = []

        for name, model in self.base_models:
            print(f"\nTraining {name}...")

            if name == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            elif name == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif name == 'lightgbm':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         callbacks=[lgb.early_stopping(50, verbose=False)])

            # 베이스 모델의 예측값 저장
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            base_train_predictions.append(train_pred)
            base_val_predictions.append(val_pred)

            print(f"  ✓ {name} training completed")

        # 베이스 모델들의 예측값을 메타 피처로 변환
        print("\n" + "="*60)
        print("Creating meta features from base model predictions...")
        print("="*60)

        X_train_meta = np.column_stack(base_train_predictions)
        X_val_meta = np.column_stack(base_val_predictions)

        print(f"Meta feature shape - Train: {X_train_meta.shape}, Val: {X_val_meta.shape}")

        # 2단계: 메타 모델 학습
        print("\n" + "="*60)
        print("STAGE 2: Training Meta Model (XGBoost)")
        print("="*60)

        verbose_flag = 100 if self.config.model.verbose > 0 else 0

        self.meta_model.fit(
            X_train_meta, y_train,
            eval_set=[(X_train_meta, y_train), (X_val_meta, y_val)],
            verbose=verbose_flag
        )

        # 학습 히스토리 저장
        results = self.meta_model.evals_result()
        if self.config.model.task == 'regression':
            self.train_history['train_loss'] = results['validation_0'].get('rmse', [])
            self.train_history['val_loss'] = results['validation_1'].get('rmse', [])
        else:
            self.train_history['train_loss'] = results['validation_0'].get('logloss', [])
            self.train_history['val_loss'] = results['validation_1'].get('logloss', [])

        print("\n" + "="*60)
        print("Stacking Ensemble Training Completed!")
        print("="*60)

    def predict(self, X):
        """스태킹 예측: 베이스 모델들의 예측 -> 메타 모델로 최종 예측"""
        X_processed = self._preprocess_data(X)

        if hasattr(self, 'feature_names_clean'):
            X_processed.columns = self.feature_names_clean

        # 1단계: 베이스 모델들의 예측값 생성
        base_predictions = []
        for _, model in self.base_models:
            pred = model.predict(X_processed)
            base_predictions.append(pred)

        # 메타 피처 생성
        X_meta = np.column_stack(base_predictions)

        # 2단계: 메타 모델로 최종 예측
        if self.config.model.task == 'regression':
            predictions = self.meta_model.predict(X_meta)
            return predictions, None
        else:
            predictions = self.meta_model.predict(X_meta)
            probabilities = self.meta_model.predict_proba(X_meta) if hasattr(self.meta_model, 'predict_proba') else None
            return predictions, probabilities

    def evaluate_model(self, X_test, y_test):
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
            print("Model Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")

        return metrics, predictions, probabilities

    def get_feature_importance(self, X):
        """메타 모델의 feature importance 반환 (베이스 모델들의 중요도)"""
        if self.meta_model is None:
            return None

        # 메타 모델의 feature importance (각 베이스 모델의 기여도)
        importance = self.meta_model.feature_importances_
        return importance

    def plot_training_history(self):
        if not os.path.exists(self.config.ui.plots_save_dir):
            os.makedirs(self.config.ui.plots_save_dir)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if self.train_history['train_loss'] and self.train_history['val_loss']:
            ax1.plot(self.train_history['train_loss'], label='Training Loss')
            ax1.plot(self.train_history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.legend()

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
            plt.savefig(f"{self.config.ui.plots_save_dir}/xgboost_stacking_results.png", dpi=300, bbox_inches='tight')

        if self.config.ui.show_plots_popup:
            plt.show()
        else:
            plt.close()

        return fig

    def save_model(self, save_dir="./models"):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, "stacking_model.pkl")
        metadata_path = os.path.join(save_dir, "stacking_metadata.pkl")

        if self.meta_model is None or not self.base_models:
            print("Warning: No model to save!")
            return False

        try:
            # 베이스 모델들과 메타 모델 모두 저장
            models_dict = {
                'base_models': self.base_models,
                'meta_model': self.meta_model
            }

            with open(model_path, 'wb') as f:
                pickle.dump(models_dict, f)

            metadata = {
                'train_history': self.train_history,
                'metrics_history': self.metrics_history,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
                'feature_names_clean': self.feature_names_clean if hasattr(self, 'feature_names_clean') else None,
            }

            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)

            print(f"Stacking model saved to {model_path}")
            print(f"  - Base models: {len(self.base_models)}")
            print(f"  - Meta model: XGBoost")
            print(f"Metadata saved to {metadata_path}")
            return True

        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, save_dir="./models"):
        model_path = os.path.join(save_dir, "stacking_model.pkl")
        metadata_path = os.path.join(save_dir, "stacking_metadata.pkl")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"Error: Model files not found in {save_dir}")
            return False

        try:
            # 베이스 모델들과 메타 모델 모두 로드
            with open(model_path, 'rb') as f:
                models_dict = pickle.load(f)

            self.base_models = models_dict['base_models']
            self.meta_model = models_dict['meta_model']

            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.train_history = metadata['train_history']
            self.metrics_history = metadata['metrics_history']
            self.label_encoders = metadata.get('label_encoders', {})
            self.feature_names = metadata.get('feature_names', None)
            self.feature_names_clean = metadata.get('feature_names_clean', None)

            print(f"Stacking model loaded from {model_path}")
            print(f"  - Base models: {len(self.base_models)}")
            print(f"  - Meta model: XGBoost")
            print(f"Metadata loaded from {metadata_path}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def model_exists(self, save_dir="./models"):
        model_path = os.path.join(save_dir, "stacking_model.pkl")
        metadata_path = os.path.join(save_dir, "stacking_metadata.pkl")
        return os.path.exists(model_path) and os.path.exists(metadata_path)
