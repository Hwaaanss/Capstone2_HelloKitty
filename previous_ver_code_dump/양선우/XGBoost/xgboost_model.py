import xgboost as xgb
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


class XGBoostWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.train_history = {'train_loss': [], 'val_loss': []}
        self.metrics_history = {}
        self.model_save_path = "xgboost_model.json"
        self.metadata_save_path = "xgboost_metadata.pkl"
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
                    print(f"🔢 Converted {col} to numeric")
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

        # Feature names 저장
        self.feature_names = X_train_processed.columns.tolist()

        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test

    def build_model(self, input_dim, output_dim=1):
        """XGBoost 모델 빌드"""
        # XGBoost는 MPS를 지원하지 않으므로 CPU 사용
        device = self.config.model.xgb_device
        if device == 'mps':
            device = 'cpu'
            print("Warning: XGBoost does not support MPS. Using CPU instead.")

        # 공통 하이퍼파라미터
        common_params = {
            'n_estimators': self.config.model.epochs,
            'learning_rate': self.config.model.lr,
            'max_depth': self.config.model.xgb_max_depth,
            'min_child_weight': self.config.model.xgb_min_child_weight,
            'subsample': self.config.model.xgb_subsample,
            'colsample_bytree': self.config.model.xgb_colsample_bytree,
            'gamma': self.config.model.xgb_gamma,
            'reg_alpha': self.config.model.xgb_reg_alpha,
            'reg_lambda': self.config.model.xgb_reg_lambda,
            'random_state': 42,
            'tree_method': self.config.model.xgb_tree_method,
            'device': device,
        }

        if self.config.model.task == 'regression':
            self.model = xgb.XGBRegressor(
                **common_params,
                eval_metric=self.config.model.xgb_eval_metric
            )
        else:
            self.model = xgb.XGBClassifier(
                **common_params,
                scale_pos_weight=self.config.model.xgb_scale_pos_weight,
                eval_metric='logloss' if self.config.model.xgb_eval_metric == 'rmse' else self.config.model.xgb_eval_metric
            )

        print(f"XGBoost model built for {self.config.model.task}")
        print(f"Hyperparameters: n_estimators={self.config.model.epochs}, lr={self.config.model.lr}, "
              f"max_depth={self.config.model.xgb_max_depth}, subsample={self.config.model.xgb_subsample}")

    def train_model(self, X_train, y_train, X_val, y_val):
        """XGBoost 모델 학습"""
        # y를 1차원으로 변환
        if len(y_train.shape) == 2 and y_train.shape[1] == 1:
            y_train = y_train.flatten()
            y_val = y_val.flatten()

        # 학습
        eval_set = [(X_train, y_train), (X_val, y_val)]

        print("Training XGBoost model...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=self.config.model.verbose
        )

        # 학습 히스토리 저장
        results = self.model.evals_result()
        if self.config.model.task == 'regression':
            self.train_history['train_loss'] = results['validation_0']['rmse']
            self.train_history['val_loss'] = results['validation_1']['rmse']
        else:
            self.train_history['train_loss'] = results['validation_0']['logloss']
            self.train_history['val_loss'] = results['validation_1']['logloss']

    def predict(self, X):
        """예측 수행"""
        X_processed = self._preprocess_data(X)

        if self.config.model.task == 'regression':
            predictions = self.model.predict(X_processed)
            return predictions, None
        else:
            predictions = self.model.predict(X_processed)
            probabilities = self.model.predict_proba(X_processed)
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
        """Feature importance 계산"""
        if self.model is None:
            return None

        importance = self.model.feature_importances_
        return importance

    def plot_training_history(self):
        """학습 히스토리 플롯"""
        if not os.path.exists(self.config.ui.plots_save_dir):
            os.makedirs(self.config.ui.plots_save_dir)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 학습 곡선
        ax1.plot(self.train_history['train_loss'], label='Training Loss')
        ax1.plot(self.train_history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
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
            plt.savefig(f"{self.config.ui.plots_save_dir}/xgboost_training_results.png", dpi=300, bbox_inches='tight')

        if self.config.ui.show_plots_popup:
            plt.show()
        else:
            plt.close()

        return fig

    def save_model(self, save_dir="./models"):
        """모델 저장"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, "xgboost_model.json")
        metadata_path = os.path.join(save_dir, "xgboost_metadata.pkl")

        if self.model is None:
            print("Warning: No model to save!")
            return False

        try:
            # XGBoost 모델 저장
            self.model.save_model(model_path)

            # 메타데이터 저장
            metadata = {
                'train_history': self.train_history,
                'metrics_history': self.metrics_history,
                'label_encoders': self.label_encoders,
                'feature_names': self.feature_names,
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
        model_path = os.path.join(save_dir, "xgboost_model.json")
        metadata_path = os.path.join(save_dir, "xgboost_metadata.pkl")

        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            print(f"Error: Model files not found in {save_dir}")
            return False

        try:
            # 메타데이터 로드
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            self.train_history = metadata['train_history']
            self.metrics_history = metadata['metrics_history']
            self.label_encoders = metadata.get('label_encoders', {})
            self.feature_names = metadata.get('feature_names', None)

            # 모델 재구축 및 로드
            if self.config.model.task == 'regression':
                self.model = xgb.XGBRegressor()
            else:
                self.model = xgb.XGBClassifier()

            self.model.load_model(model_path)

            print(f"Model loaded from {model_path}")
            print(f"Metadata loaded from {metadata_path}")
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def model_exists(self, save_dir="./models"):
        """모델 파일 존재 여부 확인"""
        model_path = os.path.join(save_dir, "xgboost_model.json")
        metadata_path = os.path.join(save_dir, "xgboost_metadata.pkl")
        return os.path.exists(model_path) and os.path.exists(metadata_path)
