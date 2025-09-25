#!/usr/bin/env python3

import pandas as pd
from utils import load_csv_with_encoding, print_dataset_info, suggest_target_columns

def debug_data_loading():
    print("🐛 Data Loading Debug Script")
    print("=" * 50)

    # 데이터 로드 시도
    data_path = './dataset/BasicData_Prep.csv'

    try:
        df = load_csv_with_encoding(data_path)
        print_dataset_info(df)

        # 타겟 컬럼 제안
        suggested = suggest_target_columns(df)
        print(f"\n🎯 Suggested target columns: {suggested}")

        # 문제가 될 수 있는 컬럼들 찾기
        print(f"\n⚠️ Potential problematic columns:")

        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(10).tolist()
                unique_count = df[col].nunique()
                unique_ratio = unique_count / len(df)

                print(f"\n📊 {col}:")
                print(f"   Type: {df[col].dtype}")
                print(f"   Unique values: {unique_count} ({unique_ratio:.2%})")
                print(f"   Sample values: {sample_values}")

                # ID 같은 컬럼 감지
                if unique_ratio > 0.9 and unique_count > 100:
                    print(f"   ⚠️ Warning: Might be ID column (too many unique values)")

                # 숫자 문자열 감지
                numeric_convertible = 0
                for val in df[col].dropna().head(100):
                    try:
                        float(str(val))
                        numeric_convertible += 1
                    except:
                        pass

                if numeric_convertible > 50:
                    print(f"Might be convertible to numeric ({numeric_convertible}/100 samples)")

        return df

    except Exception as e:
        print(f" Error: {e}")
        return None

if __name__ == "__main__":
    debug_data_loading()