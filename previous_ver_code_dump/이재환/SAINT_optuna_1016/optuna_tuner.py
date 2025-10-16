#!/usr/bin/env python3

import optuna
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import json
import os
from datetime import datetime

from config import Config, ModelConfig
from saint_model import SAINTWrapper


class OptunaSAINTTuner:
    """Optuna를 사용한 SAINT 하이퍼파라미터 최적화"""

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
            'embedding_size': trial.suggest_categorical('embedding_size', [16, 32, 64, 128]),
            'transformer_depth': trial.suggest_int('transformer_depth', 2, 8),
            'attention_heads': trial.suggest_categorical('attention_heads', [2, 4, 8, 16]),
            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.5),
            'ff_dropout': trial.suggest_float('ff_dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
        }

        # Config 업데이트
        temp_config = Config()
        temp_config.model.embedding_size = params['embedding_size']
        temp_config.model.transformer_depth = params['transformer_depth']
        temp_config.model.attention_heads = params['attention_heads']
        temp_config.model.attention_dropout = params['attention_dropout']
        temp_config.model.ff_dropout = params['ff_dropout']
        temp_config.model.lr = params['learning_rate']
        temp_config.model.batch_size = params['batch_size']
        temp_config.model.epochs = 50  # 빠른 튜닝을 위해 epoch 수 제한
        temp_config.model.task = self.task
        temp_config.model.verbose = 0
        temp_config.model.early_stopping = True
        temp_config.model.early_stopping_patience = 10

        # 모델 생성 및 학습
        try:
            model = SAINTWrapper(temp_config)
            output_dim = 1 if self.task == 'regression' else len(set(self.y_train))
            model.build_model(self.X_train.shape[1], output_dim)

            # 학습
            model.train_model(self.X_train, self.y_train.reshape(-1, 1),
                            self.X_val, self.y_val.reshape(-1, 1))

            # 검증 성능 평가
            _, predictions, _ = model.evaluate_model(self.X_val, self.y_val.reshape(-1, 1))

            if self.task == 'regression':
                # RMSE를 최소화
                mse = mean_squared_error(self.y_val, predictions)
                score = np.sqrt(mse)
            else:
                # Accuracy를 최대화하기 위해 -accuracy 반환
                score = -accuracy_score(self.y_val, predictions)

        except Exception as e:
            print(f"Trial failed: {str(e)}")
            score = float('inf') if self.task == 'regression' else 0.0

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
        print("Starting Optuna Hyperparameter Optimization for SAINT")
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
            study_name=f'saint_{self.task}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
        config.model.embedding_size = self.best_params.get('embedding_size', config.model.embedding_size)
        config.model.transformer_depth = self.best_params.get('transformer_depth', config.model.transformer_depth)
        config.model.attention_heads = self.best_params.get('attention_heads', config.model.attention_heads)
        config.model.attention_dropout = self.best_params.get('attention_dropout', config.model.attention_dropout)
        config.model.ff_dropout = self.best_params.get('ff_dropout', config.model.ff_dropout)
        config.model.lr = self.best_params.get('learning_rate', config.model.lr)
        config.model.batch_size = self.best_params.get('batch_size', config.model.batch_size)

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

    # SAINTWrapper로 데이터 전처리
    model = SAINTWrapper(config)
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(df, target_columns)

    # Optuna 튜닝 실행
    tuner = OptunaSAINTTuner(
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

    parser = argparse.ArgumentParser(description='Optuna SAINT Hyperparameter Tuning')
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
