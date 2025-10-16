#!/usr/bin/env python3

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json
import os
from datetime import datetime

from config import Config, ModelConfig
from stacking_model import StackingEnsembleWrapper


class OptunaStackingTuner:
    """Optuna를 사용한 Stacking Ensemble 하이퍼파라미터 최적화"""

    def __init__(self, config: Config, X_train, y_train, X_val, y_val, task='regression'):
        self.config = config
        self.X_train = X_train
        self.y_train = y_train.flatten() if len(y_train.shape) > 1 else y_train
        self.X_val = X_val
        self.y_val = y_val.flatten() if len(y_val.shape) > 1 else y_val
        self.task = task
        self.best_params = None
        self.study = None

    def objective(self, trial):
        """Optuna objective function"""

        # 하이퍼파라미터 탐색 범위 설정
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.5, 10.0),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'tree_method': self.config.model.xgb_tree_method,
            'device': self.config.model.xgb_device if self.config.model.xgb_device != 'mps' else 'cpu',
        }

        # Task에 따라 모델 생성
        if self.task == 'regression':
            params['eval_metric'] = 'rmse'
            model = xgb.XGBRegressor(**params)
        else:
            params['eval_metric'] = 'logloss'
            params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.5, 5.0)
            model = xgb.XGBClassifier(**params)

        # 모델 학습
        eval_set = [(self.X_train, self.y_train), (self.X_val, self.y_val)]
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=eval_set,
            verbose=False
        )

        # 검증 성능 평가
        y_pred = model.predict(self.X_val)

        if self.task == 'regression':
            # RMSE를 최소화 (optuna는 기본적으로 최소화)
            mse = mean_squared_error(self.y_val, y_pred)
            score = np.sqrt(mse)  # RMSE
        else:
            # Accuracy를 최대화하기 위해 -accuracy 반환
            score = -accuracy_score(self.y_val, y_pred)

        return score

    def tune(self, n_trials=100, timeout=None, show_progress_bar=True):
        """
        하이퍼파라미터 튜닝 실행

        Args:
            n_trials: 시도할 trial 수
            timeout: 최대 실행 시간 (초)
            show_progress_bar: 진행 상황 표시 여부

        Returns:
            최적의 하이퍼파라미터
        """
        print("=" * 60)
        print("Starting Optuna Hyperparameter Optimization")
        print("=" * 60)
        print(f"Task: {self.task}")
        print(f"Number of trials: {n_trials}")
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Validation data shape: {self.X_val.shape}")
        print("-" * 60)

        # Optuna study 생성
        direction = 'minimize' if self.task == 'regression' else 'minimize'
        self.study = optuna.create_study(
            direction=direction,
            study_name=f'stacking_{self.task}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )

        # 최적화 실행
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=show_progress_bar
        )

        # 최적 파라미터 저장
        self.best_params = self.study.best_params

        print("\n" + "=" * 60)
        print("Optimization Completed!")
        print("=" * 60)
        print(f"Best trial number: {self.study.best_trial.number}")
        print(f"Best score: {self.study.best_value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print("=" * 60)

        return self.best_params

    def get_best_params(self):
        """최적의 파라미터 반환"""
        return self.best_params

    def save_results(self, output_dir="./optuna_results"):
        """튜닝 결과 저장"""
        if self.study is None:
            print("No study to save. Please run tune() first.")
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Best parameters를 JSON으로 저장
        params_file = os.path.join(output_dir, f"best_params_{timestamp}.json")
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=4, ensure_ascii=False)
        print(f"\n✓ Best parameters saved to: {params_file}")

        # 2. Study history를 CSV로 저장
        df = self.study.trials_dataframe()
        history_file = os.path.join(output_dir, f"study_history_{timestamp}.csv")
        df.to_csv(history_file, index=False, encoding='utf-8')
        print(f"✓ Study history saved to: {history_file}")

        # 3. Visualization 저장 (optuna 시각화)
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

            # Optimization history
            fig1 = plot_optimization_history(self.study)
            fig1.figure.savefig(
                os.path.join(output_dir, f"optimization_history_{timestamp}.png"),
                dpi=300,
                bbox_inches='tight'
            )
            print(f"✓ Optimization history plot saved")

            # Parameter importances
            try:
                fig2 = plot_param_importances(self.study)
                fig2.figure.savefig(
                    os.path.join(output_dir, f"param_importances_{timestamp}.png"),
                    dpi=300,
                    bbox_inches='tight'
                )
                print(f"✓ Parameter importance plot saved")
            except:
                print("  (Parameter importance plot skipped - requires more trials)")

            plt.close('all')

        except Exception as e:
            print(f"Warning: Could not save visualization: {str(e)}")

    def update_config(self, config: Config) -> Config:
        """
        최적의 파라미터로 config 업데이트

        Args:
            config: 업데이트할 Config 객체

        Returns:
            업데이트된 Config 객체
        """
        if self.best_params is None:
            print("No best parameters found. Please run tune() first.")
            return config

        # ModelConfig 업데이트
        config.model.epochs = self.best_params.get('n_estimators', config.model.epochs)
        config.model.lr = self.best_params.get('learning_rate', config.model.lr)
        config.model.xgb_max_depth = self.best_params.get('max_depth', config.model.xgb_max_depth)
        config.model.xgb_min_child_weight = self.best_params.get('min_child_weight', config.model.xgb_min_child_weight)
        config.model.xgb_subsample = self.best_params.get('subsample', config.model.xgb_subsample)
        config.model.xgb_colsample_bytree = self.best_params.get('colsample_bytree', config.model.xgb_colsample_bytree)
        config.model.xgb_gamma = self.best_params.get('gamma', config.model.xgb_gamma)
        config.model.xgb_reg_alpha = self.best_params.get('reg_alpha', config.model.xgb_reg_alpha)
        config.model.xgb_reg_lambda = self.best_params.get('reg_lambda', config.model.xgb_reg_lambda)

        if 'scale_pos_weight' in self.best_params:
            config.model.xgb_scale_pos_weight = self.best_params['scale_pos_weight']

        print("\n✓ Config updated with best parameters")
        return config


def run_optuna_tuning(data_path, target_columns, config_path=None, n_trials=100, timeout=None):
    """
    Optuna 튜닝 실행 함수

    Args:
        data_path: 데이터 파일 경로
        target_columns: 타겟 컬럼 리스트
        config_path: config 파일 경로
        n_trials: Optuna trial 수
        timeout: 최대 실행 시간 (초)

    Returns:
        튜닝된 config와 최적 파라미터
    """
    from utils import load_csv_with_encoding

    # Config 로드
    if config_path:
        config = Config.load(config_path)
    else:
        config = Config.load() if os.path.exists('config.json') else Config()

    # 데이터 로드
    print(f"Loading data from: {data_path}")
    df = load_csv_with_encoding(data_path)
    print(f"Data shape: {df.shape}")

    # StackingEnsembleWrapper로 데이터 전처리
    model = StackingEnsembleWrapper(config)
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(df, target_columns)

    # Optuna 튜닝 실행
    tuner = OptunaStackingTuner(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        task=config.model.task
    )

    best_params = tuner.tune(n_trials=n_trials, timeout=timeout)

    # 결과 저장
    tuner.save_results()

    # Config 업데이트 및 저장
    updated_config = tuner.update_config(config)
    config_save_path = "config_optimized.json"
    updated_config.save(config_save_path)
    print(f"\n✓ Optimized config saved to: {config_save_path}")

    return updated_config, best_params


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Optuna Stacking Ensemble Hyperparameter Tuning')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, nargs='+', required=True, help='Target column names')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, help='Maximum time in seconds')

    args = parser.parse_args()

    try:
        updated_config, best_params = run_optuna_tuning(
            data_path=args.data,
            target_columns=args.target,
            config_path=args.config,
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        print("\n✓ Hyperparameter tuning completed successfully!")

    except Exception as e:
        print(f"\n✗ Tuning failed: {str(e)}")
        import traceback
        traceback.print_exc()
