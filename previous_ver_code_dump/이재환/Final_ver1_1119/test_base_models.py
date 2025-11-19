#!/usr/bin/env python3
"""
각 베이스 모델의 성능을 비교하고 가장 좋은 모델의 feature importance를 저장하는 스크립트
"""

import pickle
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score
from config import Config
from utils import load_csv_with_encoding, get_latest_preprocessed_data

def evaluate_base_models():
    print("="*60)
    print("Base Model Performance Evaluation")
    print("="*60)

    # 설정 로드
    config = Config.load()

    # 모델 로드
    print("\nLoading saved models...")
    with open('models/stacking_model.pkl', 'rb') as f:
        models_dict = pickle.load(f)

    with open('models/stacking_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    base_models = models_dict['base_models']
    feature_names = metadata.get('feature_names', [])

    print(f"Loaded {len(base_models)} base models")
    print(f"Feature count: {len(feature_names)}")

    # 데이터 로드
    print("\nLoading latest preprocessed data...")
    data_path = get_latest_preprocessed_data("./dataset")
    if not data_path:
        print("Error: No preprocessed data found")
        return

    print(f"Using data: {data_path}")

    # 저장된 메타데이터에서 base_model_metrics 확인
    base_model_metrics = metadata.get('base_model_metrics', {})

    print("\n" + "="*60)
    print("Base Model Performance (from training)")
    print("="*60)

    best_model_name = None
    best_rmse = float('inf')

    for name, metrics in base_model_metrics.items():
        print(f"\n{name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  - {metric_name}: {value:.4f}")
            if metric_name == 'rmse' and value < best_rmse:
                best_rmse = value
                best_model_name = name

    if not best_model_name:
        print("\nWarning: No base_model_metrics found in metadata.")
        print("This model was trained with an older version.")
        print("\nEvaluating base models on validation data...")

        # 데이터 준비
        from stacking_model import StackingEnsembleWrapper

        df = load_csv_with_encoding(data_path)
        target_columns = config.data.target_columns

        model_wrapper = StackingEnsembleWrapper(config)
        X_train, X_val, X_test, y_train, y_val, y_test = model_wrapper.prepare_data(df, target_columns)

        print(f"\nValidation set size: {X_val.shape}")

        # 각 베이스 모델 평가
        base_model_metrics = {}
        y_val_flat = y_val.flatten() if len(y_val.shape) == 2 else y_val

        for name, model in base_models:
            print(f"\nEvaluating {name}...")
            val_pred = model.predict(X_val)

            val_mse = mean_squared_error(y_val_flat, val_pred)
            val_rmse = np.sqrt(val_mse)
            val_r2 = r2_score(y_val_flat, val_pred)

            base_model_metrics[name] = {
                'mse': val_mse,
                'rmse': val_rmse,
                'r2': val_r2
            }

            print(f"  - RMSE: {val_rmse:.4f}")
            print(f"  - R2: {val_r2:.4f}")

        # 최고 성능 모델 찾기
        best_model_name = min(base_model_metrics.items(), key=lambda x: x[1]['rmse'])[0]
        best_rmse = base_model_metrics[best_model_name]['rmse']

        print("\n" + "="*60)
        print("Base Model Performance Comparison")
        print("="*60)

        for name, metrics in base_model_metrics.items():
            print(f"\n{name.upper()}:")
            for metric_name, value in metrics.items():
                print(f"  - {metric_name}: {value:.4f}")

        print(f"\n🏆 Best Base Model: {best_model_name.upper()} (RMSE: {best_rmse:.4f})")

    # 가장 좋은 모델의 feature importance 추출
    best_model = None
    for name, model in base_models:
        if name == best_model_name:
            best_model = model
            break

    if best_model is None:
        print(f"Error: Could not find model {best_model_name}")
        return

    # Feature importance 추출
    print("\n" + "="*60)
    print(f"Extracting Feature Importance from {best_model_name.upper()}")
    print("="*60)

    importance = best_model.feature_importances_

    # Feature names clean (XGBoost 호환성)
    feature_names_clean = metadata.get('feature_names_clean', feature_names)

    if len(feature_names_clean) != len(importance):
        print(f"Warning: Feature count mismatch ({len(feature_names_clean)} vs {len(importance)})")
        feature_names_clean = [f"feature_{i}" for i in range(len(importance))]

    # Feature importance 딕셔너리 생성
    importance_dict = dict(zip(feature_names_clean, importance.tolist()))

    # 중요도 순으로 정렬
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    # 상위 20개 출력
    print("\nTop 20 Features:")
    for i, (feature, imp) in enumerate(list(sorted_importance.items())[:20], 1):
        print(f"{i:2d}. {feature:30s}: {imp:.6f}")

    # JSON 파일로 저장
    output_file = "feature_importance.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_importance, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Feature importance saved to {output_file}")
    print(f"✓ Total features: {len(sorted_importance)}")
    print(f"✓ Based on best model: {best_model_name.upper()}")

if __name__ == "__main__":
    evaluate_base_models()
