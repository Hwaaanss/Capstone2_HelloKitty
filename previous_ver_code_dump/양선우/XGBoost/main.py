#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
from pathlib import Path

from config import Config, get_config
from xgboost_model import XGBoostWrapper
from utils import load_csv_with_encoding, print_dataset_info, suggest_target_columns

def run_training_pipeline(data_path, target_columns, config_path=None):
    if config_path:
        config = Config.load(config_path)
    else:
        config = get_config()

    print("Starting XGBoost Model Training Pipeline")
    print(f"Dataset: {data_path}")
    print(f"Target columns: {target_columns}")
    print("-" * 50)

    df = load_csv_with_encoding(data_path)
    print_dataset_info(df)

    # 타겟 컬럼이 데이터에 존재하는지 확인
    missing_targets = [col for col in target_columns if col not in df.columns]
    if missing_targets:
        print(f"Warning: Target columns not found: {missing_targets}")
        suggested = suggest_target_columns(df)
        if suggested:
            print(f"Suggested target columns: {suggested}")
        print(f"Available columns: {list(df.columns)}")
        return None

    model = XGBoostWrapper(config)
    print("Model initialized")

    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(df, target_columns)
    print("Data preprocessing completed")

    output_dim = 1 if config.model.task == 'regression' else len(set(y_train.flatten()))
    model.build_model(X_train.shape[1], output_dim)
    print("Model built")

    print("Training started...")
    model.train_model(X_train, y_train, X_val, y_val)
    print("Training completed")

    print("Evaluating model...")
    metrics, predictions, probabilities = model.evaluate_model(X_test, y_test)

    print("Generating plots...")
    model.plot_training_history()

    print("Saving trained model...")
    model.save_model()

    feature_importance = model.get_feature_importance(X_train)

    # Test 데이터셋의 원본 ENCODED_MCT 값들 출력 (전처리 전)
    # if 'ENCODED_MCT' in df.columns and 'TA_YM' in df.columns:
    #     # 원본 데이터에서 test 데이터 추출 (각 점포의 최신 데이터)
    #     max_ta_ym_per_store = df.groupby('ENCODED_MCT')['TA_YM'].max().reset_index()
    #     max_ta_ym_per_store.columns = ['ENCODED_MCT', 'max_TA_YM']

    #     df_with_max = df.merge(max_ta_ym_per_store, on='ENCODED_MCT')
    #     test_data_original = df_with_max[df_with_max['TA_YM'] == df_with_max['max_TA_YM']]

    #     test_store_codes = test_data_original['ENCODED_MCT'].unique().tolist()
    #     print(f"\n Test 데이터셋에 포함된 ENCODED_MCT 코드 ({len(test_store_codes)}개):")
    #     print("=" * 60)
    #     for i, code in enumerate(test_store_codes, 1):
    #         print(f"{i:2d}. {code}")
    #     print("=" * 60)
    #     print("위 코드들 중 하나를 Streamlit 챗봇에서 입력하여 마케팅 전략을 받아보세요!")

    print("\n Pipeline completed successfully!")
    return model, metrics, predictions, probabilities

def main():
    parser = argparse.ArgumentParser(description='XGBoost Model with LLM Integration')
    parser.add_argument('--mode', choices=['train', 'streamlit'], default='streamlit',
                       help='Run mode: train for CLI training, streamlit for web interface')
    parser.add_argument('--data', type=str, help='Path to CSV dataset')
    parser.add_argument('--target', type=str, nargs='+', help='Target column names')
    parser.add_argument('--config', type=str, help='Path to config file')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            print("Error: --data is required for training mode")
            parser.print_help()
            return

        if not os.path.exists(args.data):
            print(f"Error: Dataset file {args.data} not found")
            return

        # target이 지정되지 않으면 config에서 가져오기
        if args.target:
            target_columns = args.target
        else:
            config = Config.load(args.config) if args.config else get_config()
            target_columns = config.data.target_columns
            print(f"Using target columns from config: {target_columns}")

        try:
            run_training_pipeline(args.data, target_columns, args.config)
        except Exception as e:
            print(f"Training failed: {str(e)}")

    elif args.mode == 'streamlit':
        print("Starting Streamlit web interface...")
        os.system("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()